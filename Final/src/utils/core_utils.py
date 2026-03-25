from ast import Lambda
import numpy as np
import pdb
import os
import copy
import wandb
import statistics
from src.custom_optims.radam import RAdam
import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
#from models.model_ABMIL import ABMIL
#from models.model_DeepMISL import DeepMISL
#from models.model_MLPOmics import MLPOmics
#from models.model_MLPWSI import MLPWSI
#from models.model_SNNOmics import SNNOmics
#from models.model_MaskedOmics import MaskedOmics
#from models.model_MCATPathways import MCATPathways
from src.models.model_SurvPath import SurvPath

# remove comment below:  TODO
# from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom

#from models.model_TMIL import TMIL
#from models.model_motcat import MCATPathwaysMotCat
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from src.utils.general_utils import _get_split_loader, _print_network, _save_splits, save_model
from src.utils.loss_func import NLLSurvLoss

import torch.optim as optim


def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually.
    Print useful info (num fold and init databases, lenght of train and val ds), saves datasets of split an unique matrix in results directory.

    Args:
        - datasets : tuple
        - cur : Int     (num of fold)
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    # save datasets as unique matrix in directory of results
    #_save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))   ## TODO: removed
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split, val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function.
    Prints what it's doing, checks that in the args it's indicated 'nullSurvLoss' and calls it.
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss (or NLLRankSurvLoss)
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)  # loss function taken from other repository, little documentation in this one
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn


def _init_optim(args, model, lr, decay):
    r"""
    Init the optimizer 
    Get option for optimizer from the args and get corresponding one
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
        - lr : Float
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=lr, weight_decay=decay)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise NotImplementedError

    print('Done')
    return optimizer

def _init_model(args):
    r'''
    Checks type of path for omics (xena, hallmarks, combine, multi) given as args to set input dimension of omics.
    Checks for the modality:
        (omics baseline)
        - (comparison) "mlp_per_path":  aggregate the genes into pathways using the pathway composition ds
        - "omics":  multilayer perceptron MLP to handle tabular omics data
        - (comparison) "snn":  genomics self normalizing network to handle tabular omics data
        (unimodal and multimodal baseline)
        - (comparison) ["abmil_wsi", "abmil_wsi_pathways"]: Attention MIL (multiple instance learning) for the unimodal (WSI only) and multimodal setting (pathways + WSI)
        - (comparison) ["deepmisl_wsi", "deepmisl_wsi_pathways"]: DeepMISL for unimodal (WSI only) and multimodal (WSI + pathways)
        - "mlp_wsi":  no specific documentation, MLP for wsi
        - (comparison) ["transmil_wsi", "transmil_wsi_pathways"]: Attention MIL Implementation for unimodal (wsi) and multimodal (omics + wsi)
        - (comparison) "coattn":  MCAT architecture but with pathways instead of the 6 gene families
        - (comparison) "coattn_motcat":  model not present in the code
        (survpath)
        - "survpath": 
    '''
    
    print('\nInit Model...', end=' ')  
    
    # checks type of path for omics (xena, hallmarks, combine, multi) given as args
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    elif args.type_of_path == 'other':
        omics_input_dim = 4
    else:
        omics_input_dim = 0
    
    '''
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        # composition_df contains the pathways compositions
        model = MaskedOmics(**model_dict)
        # aggregate the genes into pathways and then pass through a fully connected layer to get the predictions (existing study)
        # For comparison with MLP transcriptomics only

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)
        # multilayer perceptron MLP to handle tabular omics data (existing study)
        # 3.1 b of paper?

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim, 
        }
        model = SNNOmics(**model_dict)
        # Implement a genomics self normalizing network to handle tabular omics data (existing study)
        # For Comaparison on transcriptomics only

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)
        # Implement Attention MIL (multiple instance learning) for the unimodal (WSI only) and multimodal setting (pathways + WSI), (existing study)
        # Used to compare with ABMIL, late fusion methods

    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)
        # Implements DeepMISL for unimodal (WSI only) and multimodal (WSI + pathways)  (existing study)
        # Used to compare with ASMIL, late fusion methods

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)
        # no specific documentation, MLP for wsi

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)
        # Attention MIL Implementation for unimodal (wsi) and multimodal (omics + wsi)
        # For Comparison on unimodal Histology/multimodal

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)
        # MCAT architecture but with pathways instead of the 6 gene families (existing study)
        # For comparison with early fusion methods

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        #model = MCATPathwaysMotCat(**model_dict)
        # not present in the code
        # For comparison with early fusion methods
'''
    # survpath 
    if args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes, 'wsi_projection_dim': args.wsi_projection_dim, 'omic_hidden_dim': args.encoding_layer_1_dim}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)  # for ablation studies
        else:
            model = SurvPath(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, dataset, set_type):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    ds_loader = None

    if set_type == 'train':
        ds_loader = _get_split_loader(args, dataset, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    elif set_type == 'test':  
        ds_loader = _get_split_loader(args, dataset, training=False, testing=False, batch_size=1)  # testing one does not work, useless to use it...
    elif set_type =='val':
        ds_loader = _get_split_loader(args, dataset, training=False, testing=False, batch_size=1)
    else:
        ds_loader = None

    '''
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None'''
    print('Done!')

    return ds_loader

def _extract_survival_metadata(clients = None, client= None):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data.
    Return structured array of censorship val and labels (event times) for both train and val ds.
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """
    '''
    all_censorships = []
    all_event_times = []

    for client in clients:
        train_loader = client.loader

        cens = train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy()
        all_censorships.extend(cens)

        event = train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy()
        all_event_times.extend(event)

    '''
    if client is not None:
        train_loader = client.loader
        #val_loader = client.val_loader
        #print('test extract survival metadata from one client')
        all_censorships = train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy() #get censorship vars for train, concatenate them as rows
        #print(all_censorships) 

        all_event_times =train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy()  # same for labels
        #print(all_event_times)

        #print('client censorships sizes:', all_censorships.shape)
        #print('client event times sizes:', all_event_times.shape)
    
    else:
        #print('all clients')
        all_censorships = []
        all_event_times = []

        for client in clients:
            train_loader = client.loader

            cens = train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy()
            all_censorships.extend(cens)

            event = train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy()
            all_event_times.extend(event)
        
        all_censorships = np.array(all_censorships, dtype=bool)
        all_event_times = np.array(all_event_times, dtype=float)
        #print('all censorships:', all_censorships)
        #print('all event times:', all_event_times)

        #print('all censorships sizes:', all_censorships.shape)
        #print('all event times sizes:', all_event_times.shape)
    
    '''
    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(), #get censorship vars for train, concatenate them as rows 
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)  # axis 0 - rows

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),  # same for labels
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)
    '''

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times) # Surv: library of sklearn to do survival analysis; Create structured array
    #print('all survival:', all_survival)
    #print('all survival sizes:', all_survival.shape)
    return all_survival



def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    Add wsi and omics data to devide. Unpack data in 'data' (from loader): y_disc, event_time, censor, clinical_data_list, mask (from data[6]) and add it to devide
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:   # unimodal, genomics
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:  # unimodal, wsi / multimodal. Both comparison
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat"]:  # just comparison 
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["survpath"]:  ## our model
        
        data_WSI = data[0].to(device) if data[0] is not None else None

        data_omics = []
        if data[1] == 0:   # if omics data is missing
            data_omics = []
        else:
            for item in data[1][0]:
                data_omics.append(item.to(device))
        
        if data[6] is None or (data[6] is not None and data[6][0,0] == 1):     # what is here?
            mask = None
        else:
            mask = data[6].to(device)
        #print('y disc: ', data[2], '- event time - censor: ', data[3], data[4], 'clinical data list: ', data[5])
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask


def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    Extracts data from Loader calling specific fuction, calls model on the data. Returns output of the model and other info retrieved from the loader.
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple 
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    # extract data indicated below from the loader, and add it to the device - returns two times mask (that may be None)
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

    if modality in ["coattn", "coattn_motcat"]:   # just comparison
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  

    elif modality == 'survpath':   ## our model

        # if no wsi present: x_path = 0
        #input_args['is_wsi'] = True
        if data_WSI is None:
            input_args = {"x_path": None}
            input_args['is_wsi'] = False
        else:
            input_args = {"x_path": data_WSI.to(device)} 
            input_args['is_wsi'] = True

        # if no omics present: is_omics = False, and no x_omic-i given
        if data_omics is None:
              input_args['is_omics'] = False
        else:
            for i in range(len(data_omics)):   # takes only omics data
                input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device) 
            input_args['is_omics'] = True 
        
        input_args['device'] = device
        
        input_args["return_attn"] = False
        out = model(**input_args)   # gives both wsi and omics data to the model - forward function is called
        
    else:   # can be ignored
        out = model(
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)   # Returns a new tensor with a dimension of size one inserted at the specified position (row! everything is in same row). The returned tensor shares the same underlying data with this tensor
    return out, y_disc, event_time, censor, clinical_data_list 


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient.
    We build a classifier such that each output logit predicted by the network correspond to a time interval.
    we define the discrete hazard function fhazard(yj|x ̄Att) = S(yˆj) where S is the sigmoid activation: fhazard(yj| ̄xAtt) represents the probability that the patient dies during time interval (tj−1, tj).
    we define the discrete survival function fsurv(yj|x ̄Att) =  productur from k=1 to h of (1 − fhazard(yk|x ̄Att)) that represents the probability that the patient survives up to time interval (tj−1, tj).
    These enable us to define the negative log-likelihood (NLL) survival loss, which generalizes NLL to data with censorship.
    by taking the negative of the sum of all logits, we can define a patient-level risk used to identify different risk groups and stratify patients.

    Args: 
        - h : torch.Tensor (output of the model for specific epoch and data)
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()



def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values. Add new risk scores, new censorship info, new event times info, clinical data info collected by the batch of specific epoch
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data


def _train_loop_survival(epoch, model, modality, loader, optimizer, loss_fn, global_weights=None, client_id=None, mu=None, method=None):
    r"""
    Perform one epoch of training. 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
        - client_id : Int
        - mu : Float
        - method : String
    
    Returns:
        - c_index : Float
        - total_loss : Float   (loss of one epoch. loss given by sum of all loss of single batches, divided by lenght of ds)
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # Set the module in training mode.

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    # one epoch
    # return (patch_features, omic_list, label, event_time, c, clinical_data, mask) - returned by the __get_item__ of the dataset by the loader
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()  # resets the gradient. clearing them ensures that each optimization step is based only on the current batch of data, preventing incorrect updates.

        # h is the output of the model on the data indicated, the rest is the data from loader unpacked
        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
        
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)  # NLLSurvLoss - uses info from label data and output from model
        loss = loss / y_disc.shape[0] # divide by num of samples

        # if args.fed_method == 'fedprox', added the fedprox loss, else loss is just returned
        loss = add_fedprox_loss(loss, mu, model, global_weights, method)
        loss_value = loss.item()  # Returns the value of this tensor as a standard Python number
        
        risk, _ = _calculate_risk(h)  # get the risk for the patients (calculate survival), and detach from gpu the survival values

        # add new info collected during batch to general array of epoch
        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        # sum all loss
        total_loss += loss_value 

        # optimize
        loss.backward()
        optimizer.step()
        #scheduler.step()

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    # when the batches of epoch are over:
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0) 
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    wandb.log({
        "train_risk_c" + str(client_id): statistics.fmean(all_risk_scores)
    })
    # calculate c-index: concordance_index_censored of the epoch
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    # from sklearn: The concordance index is defined as the proportion of all comparable pairs in which the predictions and outcomes are concordant.
    # Two samples are comparable if (i) both of them experienced an event (at different times), or (ii) the one with a shorter observed survival time experienced an event, 
    # in which case the event-free subject “outlived” the other. 
    # A pair is not comparable if they experienced events at the same time.
    # Concordance intuitively means that two samples were ordered correctly by the model. 
    # More specifically, two samples are concordant, if the one with a higher estimated risk score has a shorter actual survival time. 
    # When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count of concordant pairs.
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss


def add_fedprox_loss(loss, mu=None, model=None, global_weights=None, method=None):
    r'''
    Modifies the loss according to fedprox if the fed modality is active

    Args:
        - loss
        - mu : Float
        - model : pytorch.Model
        - method : String
    '''
    # FedProx proximal term: (mu/2) * ||w - w_global||^2

    # the fedprox method was not selected
    if method is not None and method != 'fedprox':
        return loss
    
    if global_weights is None or mu is None or mu <= 0.0:
        print('You are using the fedprox code without the correct arguments')
        return loss

    prox_reg = 0.0
    # Iterate only over trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_weights:
            gw = global_weights[name].to(param.device).type(param.dtype)
            prox_reg += torch.sum((param - gw) ** 2)

    # Proximal term = (μ / 2) * ||w - w_global||^2
    #print('pre loss:', loss.item(), ' prox_reg:', prox_reg.item(), end=' ')
    loss = loss + (mu / 2.0) * prox_reg
    #print('post loss:', loss.item(), end=';  ')

    return loss


def calculate_cindex(event_indicator, event_time, estimate, tied_tol=1e-08, n_bootstrap=9999, alpha=0.05):
    r"""
    Calculate bootstrap confidence interval for c-index and c_index
    
    Args:
        - event_indicator : np.array
        - event_time : np.array
        - estimate : np.array
        - tied_tol : Float
        - n_bootstrap : Int
        - alpha : Float

    Returns:
        - original_cindex : Float
        - lower : Float
        - upper : Float
        - bootstrap_cindexes : List of Float
    """

    lower, upper = 0.0, 0.0
    bootstrap_cindexes = []

    if n_bootstrap > 0:
        n_samples = len(event_time)
        #print(f'Calculating bootstrap c-index with {n_bootstrap} samples on {n_samples} data points...')
        bootstrap_cindexes = []
        
        for i in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            #print(f'Bootstrap indices for sample {i+1}: {indices}')
            
            boot_event = event_indicator[indices]
            boot_time = event_time[indices]
            boot_estimate = estimate[indices]
            #print(f'Bootstrap sample {i+1} - event: {boot_event}, time: {boot_time}, estimate: {boot_estimate}')
            
            # Calculate c-index for this bootstrap sample
            try:
                c_index = concordance_index_censored(boot_event, boot_time, boot_estimate, tied_tol)[0]
                bootstrap_cindexes.append(c_index)
                #print(f'Bootstrap sample {i+1}/{n_bootstrap}, c-index: {c_index:.4f}')
            except:
                # Skip if calculation fails (e.g., all censored)
                continue
        
        bootstrap_cindexes = np.array(bootstrap_cindexes)
        print(f'Total successful bootstrap samples: {len(bootstrap_cindexes)}/{n_bootstrap}')
        '''
        bootstrap_cindexes = (bootstrap_cindexes, )
        res = bootstrap(data = bootstrap_cindexes, method = 'basic')
        lower, upper = res.confidence_interval
        
        '''
        # Calculate percentile confidence interval
        lower = np.percentile(bootstrap_cindexes, 100 * alpha / 2)
        upper = np.percentile(bootstrap_cindexes, 100 * (1 - alpha / 2))
        #print(type(lower), type(upper))

    # Original c-index
    original_cindex = concordance_index_censored(event_indicator, event_time, estimate, tied_tol)[0]
    
    return original_cindex, float(lower), float(upper), bootstrap_cindexes


'''
def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics (c-index, c_index_ipcw, BS, IBS, iauc)
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc'''


def _calculate_metrics(args, test_client, clients, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores, is_cindex=False, n_boot = 9999, is_IBS=False):
    r"""
    Calculate various survival metrics (c-index, IBS)
    
    Args:
        - test_client : ClientFactory
        - clients : List of ClientFactory
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        - is_cindex : Bool
        - is_IBS : Bool
        
    Returns:
        - c_index : Float
        - lower : Float
        - upper : Float
        - bootstrap_cindexes : List of Float
        - IBS_single_list : List of Float
        - IBS_all_list : List of Float

    """
    survival_col = "survival_months_dss"
    if args.type_of_path == 'other':
        survival_col = "survival_months_os"
    data = test_client.loader.dataset.metadata[survival_col]
    bins_original = test_client.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index, lower, upper, bootstrap_cindexes = 0.0, 0.0, 0.0, []
    IBS_single_list = []
    IBS_all_list = []

    if is_cindex:
        c_index, lower, upper, bootstrap_cindexes = calculate_cindex((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08, n_bootstrap=n_boot)
    
    if is_IBS:
        IBS_all_list, IBS_single_list = calculate_IBS(clients, all_censorships, all_event_times, all_risk_by_bin_scores, which_times_to_eval_at)
    
    '''
    # other metrics present in original code, not used anymore

    c_index_ipcw, BS, iauc = 0., 0., 0., 0.
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    '''
    return c_index, lower, upper, bootstrap_cindexes, IBS_single_list, IBS_all_list


def calculate_total_loss(model, modality, val_loader, loss_fn):
    r"""
    Run a validation loop on the trained model. Return results of run, total loss, and metrics
    Not called by any function at the moment
    
    Args: 
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
    
    Returns:
        - total_loss : Float

    """
    #print('testing: val_loader', val_loader)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    #slide_ids = val_loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in val_loader:

            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["coattn", "coattn_motcat"]:  # just comparison
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                )  

            elif modality == "survpath":  ## our model
                # get data from loaders
                input_args = {"x_path": data_WSI.to(device)}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False

                input_args['device'] = device
                
                h = model(**input_args)  # get model output
                
            else:   ## other comparison
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            total_loss += loss_value
            count += 1

    total_loss /= len(val_loader.dataset)
    
    model.train()

    return total_loss


def test_model(args, model_list, loss_fn, device):
    r'''
    Runs the model for the test set and returns the metrics

    Args: 
        - args
        - model_list : list of PytorchModel
        - loss_fn : loss function
    
    Returns:
        - all_results : list of dict, results for each model in model_list
        - all_c_index : list of Float
        - all_IBS : list of Float, one for each client
        - all_loss : list of Float
    '''

    all_results = []
    all_c_index = []
    all_c_lower = []
    all_c_upper = []
    all_bootstrap_cindexes = []
    all_IBS_single = []
    all_IBS = []
    all_loss = []

    if args.fed_option != 2:
        #results_dict, test_cindex, IBS_single_list, IBS_all_list, total_loss, lower, upper, bootstrap_cindexes = _summary(args.dataset_factory.test_client, model_list[0], args.modality, loss_fn, clients = args.dataset_factory.clients)
        total_loss, results_dict, _, test_cindex, lower, upper, bootstrap_cindexes, IBS_single_list, IBS_all_list = calculate_loss_metrics(args, args.dataset_factory.test_client, model_list[0], args.modality, loss_fn, clients = args.dataset_factory.clients, is_cindex=True, is_IBS=True, is_res=True, is_risk=False)
        all_results.append(results_dict)
        all_c_index.append(test_cindex)
        all_IBS_single.extend(IBS_single_list)
        all_IBS.extend(IBS_all_list)
        all_loss.append(total_loss)
        all_c_lower.append(lower)
        all_c_upper.append(upper)
        all_bootstrap_cindexes.append(bootstrap_cindexes)

    else:
        for cid, client in enumerate(args.dataset_factory.clients):
            model_list[cid].to(device)
            #results_dict, test_cindex, IBS_single_list, IBS_all_list, total_loss, lower, upper, bootstrap_cindexes = _summary(args.dataset_factory.test_client, model_list[cid], args.modality, loss_fn, clients = [client])
            total_loss, results_dict, _, test_cindex, lower, upper, bootstrap_cindexes, IBS_single_list, IBS_all_list = calculate_loss_metrics(args, args.dataset_factory.test_client, model_list[cid], args.modality, loss_fn, clients = [client], is_cindex=True, is_IBS=True, is_res=True, is_risk=False)
            all_results.append(results_dict)
            all_c_index.append(test_cindex)
            all_IBS_single.extend(IBS_single_list)
            all_IBS.extend(IBS_all_list)
            all_loss.append(total_loss)
            all_c_lower.append(lower)
            all_c_upper.append(upper)
            all_bootstrap_cindexes.append(bootstrap_cindexes)
    
    return all_results, all_c_index, all_IBS, all_IBS_single, all_loss, all_c_lower, all_c_upper, all_bootstrap_cindexes

"""
def _summary(test_client, model, modality, loss_fn, clients, survival_train=None):
    r'''
    Run a validation loop on the trained model. Return results of run, total loss, and metrics
    
    Args: 
        - test_client : ClientFactory
        - model : Pytorch model
        - modality : String
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - IBS_list : List of Float
        - total_loss : Float
    '''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = test_client.loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in test_client.loader:
            
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            #print('y disc: ', y_disc, '- event time - censor: ', event_time, censor, 'clinical data list: ', clinical_data_list)

            if modality in ["coattn", "coattn_motcat"]:  # just comparison
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                )  

            elif modality == "survpath":  ## our model
                # get data from loaders
                input_args = {"x_path": data_WSI.to(device)}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                
                h = model(**input_args)  # get model output
                
            else:   ## other comparison
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            print('Test batch loss value:', loss_value)

            risk, risk_by_bin = _calculate_risk(h) # get risk of the patients
            print('Risk:', risk)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(test_client.loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    # put important info in new dictionary, divided by case_id (first 12 values of slide_ids)
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, IBS_single_list, IBS_all_list, lower, upper, bootstrap_cindexes = _calculate_metrics(test_client, clients, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, IBS_single_list, IBS_all_list, total_loss, lower, upper, bootstrap_cindexes
"""

def calculate_loss_metrics(args, test_client, model, modality, loss_fn, clients, is_cindex=False, n_boot=9999, is_IBS=False, is_res=False, is_risk=False):
    r'''
    For the given model, calculate the loss and the indicated metrics.
    Used for validation during training (returns val_loss and val_cindex), and for testing at the end of the training.
    Can calculate: loss, c-index (with bootstrap), IBS and the results for the final model.
    Loss is always calculated and returned.

    Args:
        - test_client : ClientFactory
        - model : Pytorch model
        - modality : String
        - loss_fn : custom loss function class
        - clients : List of ClientFactory
        - is_test : Boolean
        - is_loss : Boolean
        - is_cindex : Boolean
        - is_IBS : Boolean
        - is_res : Boolean
    
    Returns:
        - total_loss : Float
        - patient_results : dictionary
        - risk : Float
        - c_index : Float
        - lower : Float
        - upper : Float
        - bootstrap_cindexes : List of Float
        - IBS_single_list : List of Float
        - IBS_all_list : List of Float   

    '''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_case_ids = []

    case_ids = test_client.loader.dataset.metadata['case_id']
    count = 0
    with torch.no_grad():
        for data in test_client.loader:
            
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            
            if modality in ["coattn", "coattn_motcat"]:  # just comparison
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                )  

            elif modality == "survpath":  ## our model
                # if no wsi present: x_path = 0
                #input_args['is_wsi'] = True
                if data_WSI is None:
                    input_args = {"x_path": None}
                    input_args['is_wsi'] = False
                else:
                    input_args = {"x_path": data_WSI.to(device)} 
                    input_args['is_wsi'] = True

                # if no omics present: is_omics = False, and no x_omic-i given
                if data_omics is None:
                    input_args['is_omics'] = False
                else:
                    for i in range(len(data_omics)):   # takes only omics data
                        input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device) 
                    input_args['is_omics'] = True 
                
                input_args['device'] = device
                
                input_args["return_attn"] = False
                h = model(**input_args)   # gives both wsi and omics data to the model - forward function is called
                
            else:   ## other comparison
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, risk_by_bin = _calculate_risk(h) # get risk of the patients
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_case_ids.append(case_ids.values[count])
            count += 1

    total_loss /= len(test_client.loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    # return risk scores if needed
    risk = 0.0
    if is_risk:
        risk = statistics.fmean(all_risk_scores)
    
    # put important info in new dictionary, divided by case_id (first 12 values of slide_ids)
    patient_results = {}
    if is_res:
        for i in range(len(all_case_ids)):
            case_id = case_ids.values[i]
            case_id = case_id
            patient_results[case_id] = {}
            patient_results[case_id]["time"] = all_event_times[i]
            patient_results[case_id]["risk"] = all_risk_scores[i]
            patient_results[case_id]["censorship"] = all_censorships[i]
            patient_results[case_id]["clinical"] = all_clinical_data[i]
            patient_results[case_id]["logits"] = all_logits[i]
        print('lenght of patient_results: ', len(patient_results))
    
    # calculate metrics
    c_index, lower, upper, bootstrap_cindexes, IBS_single_list, IBS_all_list = _calculate_metrics(args, test_client, clients, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores, is_cindex=is_cindex, n_boot=n_boot, is_IBS=is_IBS)

    return total_loss, patient_results, risk, c_index, lower, upper, bootstrap_cindexes, IBS_single_list, IBS_all_list



def calculate_IBS(clients, all_censorships, all_event_times, all_risk_by_bin_scores, which_times_to_eval_at):
    r'''
    Calculate IBS both for single clients and for the whole dataset

    Args:
        - IBS_all_list : List of Float, empty
        - IBS_single_list : List of Float, empty
        - clients : List of ClientFactory
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        - which_times_to_eval_at : np.array

    Returns:
        - IBS_all_list : List of Float
        - IBS_single_list : List of Float

    '''
    IBS_all_list = []
    IBS_single_list = []

    survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    survival_train  = _extract_survival_metadata(clients = clients)
    try:
        ibs = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        ibs = 0.
    IBS_all_list.append(ibs)
    
    for client in clients:
        survival_train  = _extract_survival_metadata(client = client)
        try:
            ibs = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
        except:
            print('An error occured while computing IBS')
            ibs = 0.
        IBS_single_list.append(ibs) 

    return IBS_all_list, IBS_single_list

"""
def _get_lr_scheduler(args, optimizer, dataloader):
    r'''
    Present in the original code, it's not used anymore

    Learning Rate Scheduler: sophisticated mechanism to dynamically adjust this hyperparameter lr as the training progresses
    The lr increases linearly during a warmup period and then decreases. If a few epochs, not sure how it looks
    
    '''
    # get scheduler and info on epochs from args
    scheduler_name = args.lr_scheduler   # default cosine
    warmup_epochs = args.warmup_epochs   # default 1
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs  # default 2?

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)  # 1 * 106
    else:
        warmup_steps = 0

    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )

    return lr_scheduler
"""

def get_loaders(args):
    r'''
    For each client, test and val sets, init the loader and save it as client.loader

    Args:
        - args :
    
    Return:
        - None

    '''
    # get val loader
    val_dataset = args.dataset_factory.val_client.dataset
    val_loader = _init_loaders(args, val_dataset, set_type = 'val')
    args.dataset_factory.val_client.loader = val_loader
    
    # get clients loaders
    for client in args.dataset_factory.clients:
        dataset = client.dataset
        loader = _init_loaders(args, dataset, set_type = 'train')
        client.loader = loader
        # connect the val_client to each client, useful for validation during training
        client.val_client = args.dataset_factory.val_client

    # get test_loader
    test_dataset = args.dataset_factory.test_client.dataset
    test_loader = _init_loaders(args, test_dataset, set_type = 'test')
    args.dataset_factory.test_client.loader = test_loader

    '''
    # get the val loader
    val_dataset = args.dataset_factory.val_client.dataset
    val_loader = _init_loaders(args, val_dataset, set_type = 'val')
    args.dataset_factory.val_client.loader = val_loader

    return val_loader'''


def average_client_loss(all_clients_loss):
    r'''
    Computes the average of the losses in the list

    Args:
        - all_clients_loss : list of Float
    
    Returns:
        - fmean : Float
    '''

    return statistics.fmean(all_clients_loss)


def aggregate_scaffold(args, tot_samples, deltas_w, deltas_c, global_model, c_global, device="cpu"):
    r'''
    Aggregate weights for scaffold
    (∆x, ∆c) ← 1/|S | * ∑ over i∈S (∆yi, ∆ci)

    Args:
        - tot_samples : Int
        - deltas_w : list of torch.Tensor, flat
        - deltas_c : list of torch.Tensor, flat
        - global_model : pytorch model
        - c_global : torch.Tensor, flat
        - device : torch.device
    
    Returns:
        - global_model : pytorch model
        - c_global : torch.Tensor, flat
    
    '''
    # Aggregate model updates
    if tot_samples == 0:
        raise RuntimeError("Total samples from clients is zero.")

    mean_delta_w = sum(deltas_w) / float(tot_samples)  # num samples or num clients?
    #mean_delta_w = sum(deltas_w) / float(args.num_clients)
    # update global model params: theta_global += mean_delta_w
    theta_global = parameters_to_vector(global_model.parameters()).detach().cpu()
    theta_global = theta_global + mean_delta_w
    # write back
    vector_to_parameters(theta_global.to(device), global_model.parameters())
    global_model.to(device)

    # Aggregate control variate updates  # x ← x + ηg*∆x   # c ← c + (|S| / N) * ∆c
    mean_delta_c = torch.stack(deltas_c, dim=0).mean(dim=0)
    c_global = (c_global + mean_delta_c).detach()
    
    print('Testingggg')
    print("||c_global||", c_global.norm().item())
    print("||mean_delta_w||", mean_delta_w.norm().item())

    return global_model, c_global


def _train_loop_survival_scaffold(epoch, model, modality, loader, optimizer, loss_fn, c_global_dev, c_client_dev, lr, client_id, device="cpu"):
    r'''
    performs one epoch of training for SCAFFOLD

    Args:
        - epoch : Int
        - model : pytorch model
        - modality : String
        - loader : Pytorch dataloader
        - optimizer : Pytorch optimizer
        - loss_fn : loss function
        - c_global_dev : torch.Tensor, flat
        - c_client_dev : torch.Tensor, flat
        - lr : Float
        - client_id : Int
        - device : torch.device
    
    Returns:
        - c_index : Float
        - total_loss : Float
        - steps : Int
    '''

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # Set the module in training mode.

    total_loss = 0.
    steps = 0
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []


    for batch_idx, data in enumerate(loader):

        optimizer.zero_grad()

        # h is the output of the model on the data indicated
        # only wsi and omics data used for the model input; y_disc, event_time, censor taken from the data, for the loss calculation
        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
        
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)  # NLLSurvLoss
        loss = loss / y_disc.shape[0] # divide by num of samples
        # logits = h
        loss_value = loss.item()  # Returns the value of this tensor as a standard Python number
        
        risk, _ = _calculate_risk(h)  # get the risk for the patients (calculate survival), and detach from gpu the survival values

        # add new info collected during batch to general array of epoch
        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        # optimize
        loss.backward()

        #yi ← yi − ηl (gi(yi) − ci + c)
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                    
                numel = p.numel()
                # Get gradient for this parameter
                g_flat = p.grad.view(-1)
                # Get corresponding control variates
                c_i_piece = c_client_dev[idx: idx + numel]
                c_g_piece = c_global_dev[idx: idx + numel]
                
                # SCAFFOLD correction: use (grad - c_i + c_global) instead of just grad
                corrected_grad = g_flat - c_i_piece + c_g_piece
                
                # Apply update: param := param - lr * corrected_grad
                p.data.add_(-lr * corrected_grad.view_as(p))
                
                idx += numel

        # sum all loss
        optimizer.step()
        total_loss += loss_value
        steps += 1

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))


    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0) 
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    wandb.log({
        "train_risk_c" + str(client_id): statistics.fmean(all_risk_scores)
    })
    # calculate c-index: concordance_index_censored of the epoch
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))
    
    return c_index, total_loss, steps


def train_local_scaffold(args, global_model, client, c_global, c_client, lr, decay, loss_fn, device="cpu"):
    r'''
    Local training for SCAFFOLD.

    Args:
        - args
        - global_model : pytorch model
        - client : ClientFactory object
        - c_global : torch.Tensor, flat
        - c_client : torch.Tensor, flat
        - lr : Float
        - loss_fn : loss function
        - device : torch.device

    Returns:
        - client_model : pytorch model
        - client_state_dict : dictionary
        - n : Int
        - client_loss : Float
        - client_cindex : Float
        - client_risk : Float
        - delta_c : torch.Tensor, flat

    '''
    model = copy.deepcopy(global_model).to(device)

    #---> init optimizer with specific lr
    optimizer = _init_optim(args, model, lr = lr, decay = decay)

    #---> init loaders for client data
    train_loader = client.loader

    steps = 0

    # Convert control variates to device and proper dtype
    c_global_dev = c_global.to(device)
    c_client_dev = c_client.to(device)

    for epoch in range(args.max_epochs):
        print('start epoch num ', epoch)

        train_cindex, train_loss, steps_l = _train_loop_survival_scaffold(epoch, model, args.modality, train_loader, optimizer, loss_fn, c_global_dev, c_client_dev, lr, client_id=client.client_id, device=device)
        
        steps += steps_l

        wandb.log({
            "train_loss_c-" +str(client.client_id) : train_loss,
            "train_cindex_c-"+str(client.client_id) : train_cindex
        })

    # calculate loss, cindex and risk of the model on the val set
    #client_val_loss = calculate_total_loss(model, args.modality, client.val_loader, loss_fn)
    client_val_loss, _, client_val_risk, client_val_cindex, _, _, _, _, _,  = calculate_loss_metrics(args, client.val_client, model, args.modality, loss_fn, clients = [client], is_cindex = True, n_boot=0, is_risk = True)
    wandb.log({
        "client_val_loss_c" + str(client.client_id): client_val_loss,
        "client_val_cindex_c" + str(client.client_id): client_val_cindex,
        "client_val_risk_c" + str(client.client_id): client_val_risk
    })  
    print(f"Client {client.client_id}: validation loss = {client_val_loss:.4f}; c-index = {client_val_cindex:.4f}; risk = {client_val_risk:.4f}")

    # compute delta_c:            # c+i ← ci − c + 1/K_ηl (x − yi)   # ∆ci ← (c+i − ci) = − c + 1/K_ηl (x − yi)
    # c_i_new = c_i - c_global + (1 / (steps * lr)) * (theta_global - theta_i)
    # delta_c = c_i_new - c_i = -c_global + (1/(steps*lr)) * (theta_global - theta_i)
    theta_global = parameters_to_vector(global_model.parameters()).detach().cpu()
    theta_i = parameters_to_vector(model.parameters()).detach().cpu()
    delta_c = (-c_global + (1.0 / (steps * lr)) * (theta_global - theta_i)).detach().cpu()

    # delete optimizer to save up space in memory
    del optimizer
    torch.cuda.empty_cache()

    model.cpu()

    return model.state_dict(), len(client.dataset), client_val_loss, client_val_cindex, client_val_risk, delta_c



def aggregate_fedopt(args, global_model, client_states, client_sizes, server_optimizer, device, lr_client = None):
    r'''
    Aggregate the weights of the clients according to FedOpt

    Args:
        - args
        - global_model : Pytorch model
        - client_states : list of state_dict
        - client_sizes : list of Int

    Returns:
        - global_model : Pytorch model (updated global model after FedOpt)
    TODO to be tested
    '''
    scale_aggregated = True

    # global state (state_dict) BEFORE client updates
    global_state = copy.deepcopy(global_model.state_dict())

    # compute weighted aggregated new_state (same shape as state dict)
    total = sum(client_sizes)
    # initialize new_state as zeros
    new_state = {k: torch.zeros_like(v) for k, v in client_states[0].items()}
    for i in range(len(client_states)):
        weight = client_sizes[i] / total
        for k in new_state.keys():
            new_state[k] += client_states[i][k] * weight

    #print('aggregated new state: ', new_state)
    # compute aggregated_delta = sum_k p_k * (w - w_k_new)
    aggregated_delta = {}
    for k in global_state.keys():
        aggregated_delta[k] = (global_state[k].to(device) - new_state[k].to(device))

    '''
    #print('aggregated delta: ', aggregated_delta)
    # optional scaling into gradient-like magnitude -- useful for when there are a lot of batches (centralized), for federated not big changes
    if scale_aggregated:
        if lr_client is not None:
            divisor = lr_client * float(args.epochs) * float(client_sizes[0])  # last is num_steps, with batch_size = 1 is the size of the client data
            if divisor != 0:
                for k in aggregated_delta:
                    aggregated_delta[k] = aggregated_delta[k] / divisor
            else:
                # safeguard: if divisor is zero, skip scaling
                print("Warning: divisor for scaling aggregated delta is zero; skipping scaling.")
        else:
            print("Warning: scale_aggregated=True but client_lr not found in args; skipping scaling.")'''

    #print('aggregated delta after scaling: ', aggregated_delta)
    # optional clipping by global norm
    '''
    if clip_aggregated:
        # compute global norm
        total_norm_sq = 0.0
        for k in aggregated_delta:
            total_norm_sq += (aggregated_delta[k].float().norm() ** 2).item()
        total_norm = total_norm_sq ** 0.5
        if total_norm > clip_norm:
            clip_coef = clip_norm / (total_norm + 1e-12)
            for k in aggregated_delta:
                aggregated_delta[k] = aggregated_delta[k] * clip_coef'''

    # Apply aggregated_delta as "gradients" for server optimizer
    # server_optimizer was created on global_model.parameters(), so we set .grad on those params
    server_optimizer.zero_grad()

    # Map state_dict key names to model parameters
    name_to_param = {name: p for name, p in global_model.named_parameters()}

    # assign grads
    for name, param in name_to_param.items():
        if name in aggregated_delta:
            # ensure same device / dtype
            g = aggregated_delta[name].to(param.device).to(param.dtype)
            #print('grad for param', name, ': ', g, end='; ')
            # set .grad to the gradient-like tensor
            # PyTorch expects .grad to be a tensor (not Parameter)
            param.grad = g.clone().detach()
            #print('set grad: ', param.grad, end='; ')
        else:
            # If it's a buffer like running_mean/running_var they won't be in named_parameters.
            # For those cases (buffers), you can directly set them in state_dict if desired.
            pass

    # step server optimizer
    server_optimizer.step()
    # after server optimizer step, ensure model is on device
    global_model.to(device)

    return global_model


def federated_algorithm(args, global_model, clients, loss_fn, lr_to_tune=None, decay_to_tune=None, device=None):
    r'''
    According to the federated method selected, call the corresponding federated algorithm

    Args: 
        - args :
        - global_model : pytorch model
        - clients : ClientFactory list
        - loss_fn : loss function
        - device _ torch.device

    Retrun:
        - global_model : list of Pytorch model (updated global model after FedAvg or list of Client Models)
    '''

    model_list = []
    val_loss = 0.
    if args.fed_method in ['fedavg', 'fedprox']:
        print('Training with fedavg or fedprox')
        model_list, val_cindex = federated_avg(
            args=args,
            global_model = global_model,
            clients = clients,
            loss_fn = loss_fn,
            lr = lr_to_tune if lr_to_tune is not None else args.lr,
            decay = decay_to_tune if decay_to_tune is not None else args.reg,
            device = device
        )

    elif args.fed_method == 'fedopt':
        print('training with fedopt')
        model_list, val_cindex = federated_opt(
            args=args,
            global_model = global_model,
            clients = clients,
            loss_fn = loss_fn,
            lr_server = lr_to_tune if lr_to_tune is not None else args.lr_server,
            device = device
        )

    elif args.fed_method == 'scaffold':
        print('training with scaffold')
        model_list, val_cindex = federated_scaffold(
            args=args,
            global_model = global_model,
            clients = clients,
            loss_fn = loss_fn,
            lr = args.lr,
            decay = args.reg,
            device = device
        )

    else:
        print('federated method selecetd not implemented /doesn\'t exists')
    
    return model_list, val_cindex


def federated_scaffold(args, global_model, clients, loss_fn, lr, decay, device):
    r'''
    Perform Federated Scaffold on the global model using the client datasets. 
    Training goes on for max args.max_rounds times, each round every client trains on the model and the state is updated.
    Once every client trained, the global model is evaluated, and early stopping is evaluated.

    Args:
        - global_model : Pytorch model 
        - clients : List of ClientFactory
        - val_loader : DataLoader of the validation set
        - loss_fn : loss function
        - device : torch.device

    Returns:
        - global_model : list of Pytorch model (updated global model after FedAvg or list of Client Models)

    '''
    # init parameters for LRScheduler
    server_optimizer = _init_optim(args, global_model, lr=lr, decay = decay)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer,
        mode='max',
        factor=0.1,
        patience = args.lr_pat
    )

    new_lr = server_optimizer.param_groups[0]['lr']
    print('current lr: ', new_lr)

    # init early stopping / wandb etc — unchanged
    patience = args.patience
    best_val_cindex = 0.0
    patience_counter = 0
    best_round = 0
    saving_weights_path = args.model_dir
    isBestSaved = False
    wandb.log({"best_val_cindex": best_val_cindex}, step=0)

    # init scaffold
    # get the lenght for the c variables, d
    global_model.to(device)
    theta_global_vec = parameters_to_vector(global_model.parameters()).detach().cpu()
    d = theta_global_vec.numel()

    # server and clients control variate c
    c_global = torch.zeros(d, device='cpu')
    c_clients = {}
    for client in clients:
        c_clients[client.client_id] = torch.zeros(d, device='cpu')

    for r in range(args.max_rounds):

        print(f"\n### Round {r}/{args.max_rounds} ###")

        client_sizes = []
        all_clients_loss = []
        all_clients_cindex = []
        all_clients_risk = []
        current_lr = new_lr

        deltas_w = []   # will hold (raw delta vector * weight) for weighted averaging
        deltas_w_raw = []  # hold raw delta vectors (delta_theta_i = model with client train - global of previous round)
        deltas_c = []   # hold delta_c vectors (c_i_new - c_i), returned by local train
        #participating_client_ids = []

        total_samples = 0

        for client in clients:

            print('Client num:', client.client_id)
            
            w_state_dict, n, client_loss, client_cindex, client_risk, delta_c = train_local_scaffold(
                args, global_model, client,
                c_global=c_global.clone(), c_client=c_clients[client.client_id].clone(),
                lr=current_lr, decay = decay, loss_fn=loss_fn, device=device
            )

            # convert w_state_dict to flat vector (theta_i)
            tmp_model = copy.deepcopy(global_model)
            tmp_model.load_state_dict(w_state_dict)
            theta_i = parameters_to_vector(tmp_model.parameters()).detach().cpu()
            theta_global = parameters_to_vector(global_model.parameters()).detach().cpu()
            delta_theta_i = (theta_i - theta_global)

            # For weighted model averaging replicate FedAvg weighting
            deltas_w_raw.append(delta_theta_i)
            deltas_w.append(delta_theta_i * float(n))  # weight by client samples (same as FedAvg)
            deltas_c.append(delta_c)  # delta_c should be flat vector on CPU
            
            client_sizes.append(n)
            total_samples += n
            all_clients_loss.append(client_loss)
            all_clients_cindex.append(client_cindex)
            all_clients_risk.append(client_risk)

            print('End of training, client size: ', n)
            print('')

    

        global_model, c_global = aggregate_scaffold(args, tot_samples=total_samples, deltas_w=deltas_w, deltas_c=deltas_c, global_model=global_model, c_global=c_global, device= device)

        # update the c_client for each client  # c ← c + |S|/ N(∆c)  where |S|=N
        for idx, client in enumerate(clients):
            c_clients[client.client_id] = (c_clients[client.client_id] + deltas_c[idx]).detach().cpu()

        if r % 10 == 0:
            print('Testingg')
            print("||c_global||", c_global.norm().item())
            for client in clients:
                print(f"||c_client {client.client_id}||", c_clients[client.client_id].norm().item())

        # calculate average of the client losses, cindexes and risks
        model_val_loss = average_client_loss(all_clients_loss)
        model_val_cindex = average_client_loss(all_clients_cindex)
        model_val_risk = average_client_loss(all_clients_risk)
        wandb.log({
            "round_val_loss": model_val_loss,
            "round_val_cindex": model_val_cindex,
            "round_val_risk": model_val_risk
        })
        print(f"Round {r}: Validation loss of the model (mean of the clients) = {model_val_loss:.4f}")
        print(f"Round {r}: Validation c-index of the model (mean of the clients) = {model_val_cindex:.4f}")
        print(f"Round {r}: Validation risk of the model (mean of the clients) = {model_val_risk:.4f}")

        lr_scheduler.step(model_val_cindex)

        if r >= 1:
            # early stopping
            if model_val_cindex > best_val_cindex + 1e-4:
                best_val_cindex = model_val_cindex
                patience_counter = 0
                best_round = r
                torch.save(global_model.state_dict(), saving_weights_path)
                isBestSaved = True

                wandb.log({
                    "best_val_cindex": best_val_cindex
                })
                print('--- Found a lower val_loss, the model at this point is saved   ---')
            
            else:
                patience_counter += 1

            new_lr = server_optimizer.param_groups[0]['lr']
            print('current lr at the end of the round: ', new_lr)
            wandb.log({"end_of_round_lr": new_lr})

            if patience_counter >= patience:
                if isBestSaved:
                    global_model.load_state_dict(torch.load(saving_weights_path))
                    artifact = wandb.Artifact('model', type='model')
                    artifact.add_file(saving_weights_path)
                    wandb.log_artifact(artifact)
                    print('Collected the best model of previous round num ', best_round)

                print(f"Early stopping triggered at round {r}")
                wandb.log({"num_rounds_done": r})
                break

    return [global_model], best_val_cindex


def federated_opt(args, global_model, clients, loss_fn, lr_server=None, device=None):
    r'''
    Perform Federated Optimization (FedOpt) on the global model using the client datasets. 
    Training goes on for max args.max_rounds times, each round every client trains on the model and the state is updated.
    Once every client trained, the global model is evaluated, and early stopping is evaluated.

    Args:
        - global_model : Pytorch model 
        - clients : List of ClientFactory
        - val_loader : DataLoader of the validation set
        - loss_fn : loss function
        - device : torch.device

    Returns:
        - global_model : list of Pytorch model (updated global model after FedAvg or list of Client Models)

    '''
    # init server optimizer to aggregate the weights of the model
    lr_server = lr_server if lr_server is not None else args.lr_server
    server_optimizer = _init_optim(args, global_model, lr = lr_server, decay = args.reg)
    
    # init parameters for LRScheduler - the lr scheduler updates the value of the lr inside the scheduler automathically during .step()
    client_optimizer = _init_optim(args, global_model, lr = args.lr_client, decay = args.reg)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        client_optimizer,
        mode='max',
        factor=0.1,
        patience = args.lr_pat
    )
    
    # get the current lr -- the one in the args
    c_lr = client_optimizer.param_groups[0]['lr']
    print('current lr for clients: ', c_lr)
    s_lr = server_optimizer.param_groups[0]['lr']
    print('current lr for server: ', s_lr)

    # init parameters for early stopping
    patience = args.patience
    best_val_cindex = 0.0
    patience_counter = 0
    best_round = 0
    saving_weights_path = args.model_dir
    isBestSaved = False
    wandb.log({
        "best_val_cindex": best_val_cindex
    }, step=0)


    for r in range(args.max_rounds):

        print(f"\n### Round {r}/{args.max_rounds} ###")

        client_states = []
        client_sizes = []
        all_clients_loss = []
        all_clients_cindex = []
        all_clients_risk = []
        current_c_lr = c_lr

        # Local training on each client
        for client in clients:

            print('Client num:', client.client_id)
            client_model, w, n, client_loss, client_cindex, client_risk = train_local(args, global_model, client, lr=current_c_lr, decay = args.reg, loss_fn=loss_fn, device=device)
            client_states.append(w)
            client_sizes.append(n)
            all_clients_loss.append(client_loss)
            all_clients_cindex.append(client_cindex)
            all_clients_risk.append(client_risk)

            print('End of training, client size: ', n)
            print('')

        # fedopt is only in federated setting: aggregate the values of the model, update the model, 
        # calculate the loss of the model for the model, apply early stopping
        if args.fed_option != 2:    
            # Aggregate (FedOpt)
            global_model = aggregate_fedopt(args, global_model, client_states, client_sizes, server_optimizer, device, current_c_lr)

            # calculate loss of the model
            model_val_loss = average_client_loss(all_clients_loss)
            model_val_cindex = average_client_loss(all_clients_cindex)
            model_val_risk = average_client_loss(all_clients_risk)

            #val_loss = calculate_total_loss(global_model, args.modality, val_loader, loss_fn)

            wandb.log({
                "round_val_loss": model_val_loss,
                "round_val_cindex": model_val_cindex,
                "round_val_risk": model_val_risk
            })
            print(f"Round {r}: Validation loss of the model (mean of the clients) = {model_val_loss:.4f}")
            print(f"Round {r}: Validation c-index of the model (mean of the clients) = {model_val_cindex:.4f}")
            print(f"Round {r}: Validation risk of the model (mean of the clients) = {model_val_risk:.4f}")

            # update the lr scheduler - updates the lr of the server optimizer
            lr_scheduler.step(model_val_cindex)

            # Check early stopping condition
            if model_val_cindex > best_val_cindex + 1e-4:
                best_val_cindex = model_val_cindex
                patience_counter = 0
                best_round = r
                torch.save(global_model.state_dict(), saving_weights_path)
                isBestSaved = True

                wandb.log({
                    "best_val_cindex": best_val_cindex
                })
                print('--- Found a higher val_c-index, the model at this point is saved   ---')
        
            else:
                patience_counter += 1
            # else if val_loss ≤ best_val_loss + ε then: c ← 0

            # get new lr (may be updated or not)
            c_lr = client_optimizer.param_groups[0]['lr']
            print('current client lr at the end of the round: ', c_lr)
            wandb.log({
                "end_of_round_c_lr" : c_lr
            })

            if patience_counter >= patience:
                # get the model saved at best_val_cindex and return that
                #if isBestSaved:
                global_model.load_state_dict(torch.load(saving_weights_path))
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file(saving_weights_path)
                wandb.log_artifact(artifact)
                print('Collected the best model of previous round num ', best_round)

                print(f"Early stopping triggered at round {r}")
                wandb.log({
                    "num_rounds_done": r
                })
                break
    
    return [global_model], best_val_cindex


def federated_avg(args, global_model, clients, loss_fn, lr, decay, device):
    r'''
    Perform Federated Averaging (FedAvg) or FedProx on the global model using the client datasets. 
    Training goes on for max args.max_rounds times, each round every client trains on the model and the state is updated.
    Once every client trained, the global model is evaluated, and early stopping is evaluated.

    Args:
        - args : 
        - global_model : Pytorch model 
        - clients : List of ClientFactory
        - val_loader : DataLoader of the validation set
        - loss_fn : loss function
        - device : torch.device

    Returns:
        - global_model : list of Pytorch model (updated global model after FedAvg or list of Client Models)

    '''
    # init parameters for LRScheduler - the lr scheduler updates the value of the lr inside the scheduler automathically during .step()
    # parameters saved this way for use of raytune
    learning_rate = lr if lr is not None else args.lr
    reg = decay if decay is not None else args.reg
    server_optimizer = _init_optim(args, global_model, lr = learning_rate, decay = reg)
    #print('patience for lr scheduler: ', args.patience //2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        server_optimizer,
        mode = 'max',
        factor = 0.1,
        patience = args.lr_pat
    )
    
    # get the current lr -- the one in the args
    new_lr = server_optimizer.param_groups[0]['lr']
    print('current lr: ', new_lr)
    #print('lr patience: ', 20 if args.fed_option != 1 else 45 )

    # init parameters for early stopping
    patience = args.patience
    best_val_cindex = 0.0
    patience_counter = 0
    best_round = 0
    saving_weights_path = args.model_dir
    isBestSaved = False
    wandb.log({
        "best_val_cindex": best_val_cindex
    }, step=0)

    all_clients_models = []   # only for island options
    all_islands_loss = []
    all_islands_cindex = []

    for r in range(args.max_rounds):
        print(f"\n### Round {r}/{args.max_rounds-1} ###")

        client_states = []
        client_sizes = []
        all_clients_loss = []
        all_clients_cindex = []
        all_clients_risk = []
        current_lr = new_lr

        # Local training on each client
        for client in clients:

            print('Client num:', client.client_id)
            client_model, w, n, client_loss, client_cindex, client_risk = train_local(args, global_model, client, lr=current_lr, decay = reg, loss_fn=loss_fn, device=device)
            client_states.append(copy.deepcopy(w))
            client_sizes.append(copy.deepcopy(n))
            all_clients_loss.append(copy.deepcopy(client_loss))
            all_clients_cindex.append(copy.deepcopy(client_cindex))
            all_clients_risk.append(copy.deepcopy(client_risk))

            # to save up space in GPU memory, update list of models only for island option
            if args.fed_option == 2:
                client_model.cpu()
                all_clients_models.append(copy.deepcopy(client_model))
                all_islands_loss.append(copy.deepcopy(client_loss))
                all_islands_cindex.append(copy.deepcopy(client_cindex))

            print('End of training, client size: ', n)
            print('')

        # if it's federated or centralized: aggregate the values of the model, update the model, 
        # calculate the loss of the model for the model, apply early stopping
        if args.fed_option != 2:   

            # Aggregate (FedAvg) if the option is federated (or centralized, also works)
            total = sum(client_sizes)
            new_state = copy.deepcopy(client_states[0])
            for k in new_state.keys():
                new_state[k] = sum(
                    client_states[i][k] * (client_sizes[i] / total)
                    for i in range(len(client_states))
                )

            # Update global model
            global_model.load_state_dict(new_state)
            global_model.to(device)

            # calculate loss, cindex, risk of the model
            model_val_loss = average_client_loss(all_clients_loss)  # TODO change name of function
            model_val_cindex = average_client_loss(all_clients_cindex)
            model_val_risk = average_client_loss(all_clients_risk)
            
            # save and print the values for loss, cindex and risk
            wandb.log({
                "round_val_loss": model_val_loss,
                "round_val_cindex": model_val_cindex,
                "round_val_risk": model_val_risk
            })
            print(f"Round {r}: Validation loss of the model (mean of all clients) = {model_val_loss:.4f}")
            print(f"Round {r}: Validation c_index of the model (mean of all clients) = {model_val_cindex:.4f}")
            print(f"Round {r}: Validation risk of the model (mean of all clients) = {model_val_risk:.4f}")

            if r >= 4:  # don't do early stopping for the first 5 rounds
                # update the lr scheduler - updates the lr of the server optimizer
                lr_scheduler.step(model_val_cindex)

                # Check early stopping condition using cindex
                if model_val_cindex > best_val_cindex + 1e-4:
                    best_val_cindex = model_val_cindex
                    patience_counter = 0
                    best_round = r
                    torch.save(global_model.state_dict(), saving_weights_path)
                    isBestSaved = True

                    wandb.log({
                        "best_val_cindex": best_val_cindex
                    })
                    print('--- Found a higher val_cindex, the model at this point is saved   ---')
            
                else:
                    patience_counter += 1
                # else if val_loss ≤ best_val_loss + ε then: c ← 0

                # get new lr (may be updated or not)
                new_lr = server_optimizer.param_groups[0]['lr']
                wandb.log({
                    "end_of_round_lr" : new_lr
                })
                print('current lr at the end of the round: ', new_lr)

                if patience_counter >= patience:
                    # get the model saved at best_val_cindex and return that
                    #if isBestSaved:
                    global_model.load_state_dict(torch.load(saving_weights_path))
                    artifact = wandb.Artifact('model', type='model')
                    artifact.add_file(saving_weights_path)
                    wandb.log_artifact(artifact)

                    print('Collected the best model of previous round num ', best_round)

                    print(f"Early stopping triggered at round {r}")
                    wandb.log({
                        "num_rounds_done": r
                    })
                    break

    if args.fed_option == 2:

        if len(all_islands_loss) > 0:
            avg_client_loss = average_client_loss(all_islands_loss)
            print('Average client val loss on islands: ', avg_client_loss)

        if len(all_islands_cindex) > 0:
            avg_client_cindex = average_client_loss(all_islands_cindex)
            print('Average client val cindex on islands: ', avg_client_cindex)
            return all_clients_models, avg_client_cindex
        else:
            print('the cindex of the clients wasn\'t saved, something is wrong')
            return all_clients_models, None
    
    return [global_model], best_val_cindex



def train_local(args, model, client, lr, decay, loss_fn, device="cpu"):
    r"""
    Trains the model for the set number of epochs.
    
    Args:
        - args
        - model : PyTorch Model
        - client : ClientFactory
        - lr : Float
        - loss_fn : Loss function
        - device : String
        
    Returns:
        - model_state_dict : dictionary
        - len(client.dataset) : Int
        - client_val_loss : Float      
        - client_val_cindex : Float    
        - client_val_risk : Float
    """

    print('lr at the beginning of the client local training: ', lr)

    # work on a copy of the global model
    model = copy.deepcopy(model)
    model.to(device)
    global_weights = copy.deepcopy(model.state_dict())

    #---> init optimizer with specific lr
    optimizer = _init_optim(args, model, lr = lr, decay = decay)

    #---> init loaders for client data
    train_loader = client.loader

    # lr scheduler  - remove it
    #lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    client_val_cindex = 0.0
    client_val_loss = 0.0
    client_val_risk = 0.0
    
    if args.fed_option == 2:

        # lr scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,      # using the local optimizer
            mode='max',
            factor=0.1,
            patience = args.lr_pat
        )
        
        # get the current lr -- the one in the args
        new_lr = optimizer.param_groups[0]['lr']
        print('current lr_islands: ', new_lr)

        # early stopping parameters
        patience = args.patience
        best_val_cindex = 0.0
        patience_counter = 0
        best_epoch = 0
        saving_weights_path = args.model_dir
        isBestSaved = False
        wandb.log({
            "best_val_cindex_c" + str(client.client_id): best_val_cindex
        }, step=0)

    
    for epoch in range(args.max_epochs):
        
        print('start epoch num ', epoch)
       
        if args.fed_method == 'fedprox':
            train_cindex, train_loss = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, loss_fn, global_weights, client_id=client.client_id, mu=args.mu, method=args.fed_method)
        
        elif args.fed_method == 'fedavg' or args.fed_method == 'fedopt':
            train_cindex, train_loss = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, loss_fn, global_weights=None, client_id=client.client_id, mu=None, method=args.fed_method)
        
        else:
            print('The fed method indicated is not correct, something is wrong in the code')
        
        wandb.log({
            "train_loss_c-" +str(client.client_id) : train_loss,
            "train_cindex_c-"+str(client.client_id) : train_cindex
        })

        if args.fed_option == 2:   # island option
            # evaluate using val set the model at the end of the epoch
            #epoch_val_loss = calculate_total_loss(model, args.modality, client.val_loader, loss_fn)
            epoch_val_loss, _, epoch_val_risk, epoch_val_cindex, _, _, _, _, _,  = calculate_loss_metrics(args, client.val_client, model, args.modality, loss_fn, clients = [client], is_cindex = True, n_boot=0, is_risk = True)
            
            print(f'Epoch {epoch}, val loss: {epoch_val_loss}, val cindex: {epoch_val_cindex}, val risk: {epoch_val_risk}')
            wandb.log({
                "val_loss_c-" +str(client.client_id) : epoch_val_loss,
                'val_cindex_c-'+str(client.client_id) : epoch_val_cindex,
                'val_risk_c-'+str(client.client_id) : epoch_val_risk
            })

            if epoch >= 4:
                # lr scheduler update
                lr_scheduler.step(epoch_val_cindex)

                # early stopping on cindex
                if epoch_val_cindex > best_val_cindex + 1e-4:
                    best_val_cindex = epoch_val_cindex
                    patience_counter = 0
                    best_epoch = epoch
                    torch.save(model.state_dict(), saving_weights_path)
                    isBestSaved = True

                    wandb.log({
                        "best_val_cindex_c" + str(client.client_id): best_val_cindex
                    })
                    print('--- Found a higher val_c-index, the model at this point is saved   ---')
            
                else:
                    patience_counter += 1
                # else if val_loss ≤ best_val_loss + ε then: c ← 0

                if patience_counter >= patience:
                    # get the model saved at best_val_loss and return that
                    if isBestSaved:
                        model.load_state_dict(torch.load(saving_weights_path))
                        print('Collected the best model of previous epoch num ', best_epoch)
                        artifact = wandb.Artifact('model_c' + str(client.client_id), type='model')
                        artifact.add_file(saving_weights_path)
                        wandb.log_artifact(artifact)

                    client_val_cindex = best_val_cindex
                    client_val_loss = epoch_val_loss  # saving the last val loss and val risk, not the best ones - the values won't be used for islands after this moment
                    client_val_risk = epoch_val_risk
                    print(f"Early stopping triggered at epoch {epoch}")
                    wandb.log({
                        "num_epochs_done": epoch
                    })
                    break


    # delete optimizer to save up space in memory
    del optimizer
    torch.cuda.empty_cache()

    if args.fed_option != 2:
        # calculate vall loss, cindex and risk of the client
        #client_val_loss = calculate_total_loss(model, args.modality, client.val_loader, loss_fn)
        client_val_loss, _, client_val_risk, client_val_cindex, _, _, _, _, _, = calculate_loss_metrics(args, client.val_client, model, args.modality, loss_fn, clients = [client], is_cindex = True, n_boot=0, is_risk = True)
        wandb.log({
            "client_val_loss_c" + str(client.client_id): client_val_loss,
            "client_val_cindex_c" + str(client.client_id): client_val_cindex,
            "client_val_risk_c" + str(client.client_id): client_val_risk
        })  
        print(f"Client {client.client_id}: validation loss = {client_val_loss:.4f}; c-index = {client_val_cindex:.4f}; risk = {client_val_risk:.4f}")

    # get model on CPU to save up space in GPU
    model.cpu()

    # return for island 2
    return model, model.state_dict(), len(client.dataset), client_val_loss, client_val_cindex, client_val_risk


def _train_test(args, run=None, lr=None, decay=None):
    """   
    Performs training of the federated clients and testing of the final global model
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - test_cindex : Float
        - IBS_list : Float
        - total_loss : Float
    """

    print('Start of training')
    if args.fed_method == 'fedopt':
        lr = args.lr_server if lr is None else lr

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #----> init loss function, calls for NLLSurvLoss
    loss_fn = _init_loss_function(args)

    #----> init model, get model needed (if for survpath or comparison studies, according to modality)
    # some values are the same for all clients, do not depend on patients, so can be taken from first client dataset
    args.omic_names = args.dataset_factory.clients[0].omic_names
    args.omic_sizes = args.dataset_factory.clients[0].omic_sizes
    args.n_classes = args.n_classes
    model = _init_model(args)

    #print('Testinggg')
    #print(model)
    
    # used to init the loaders here, but put in the main after the scaler
    # federated averaging
    model_list, val_cindex = federated_algorithm(
        args=args,
        global_model = model,
        clients = args.dataset_factory.clients,
        loss_fn = loss_fn,
        lr_to_tune = lr if lr is not None else args.lr,
        decay_to_tune = decay if decay is not None else args.reg,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print('End of training of the global model')
    print('Testing :::: Number of models returned: ', len(model_list))

    if args.is_save_model:
        save_model(args, model_list)

    '''
    print('retrieving an already trained model')
    artifact = run.use_artifact("model:v1067", type='model') # TODO change name
    datadir = artifact.download()
    model.load_state_dict(torch.load(f"{datadir}/weights_3.pth"))  # TODO
    #model = torch.load(f"{datadir}/weights_3.pth")
    model_list = [model]
    val_loss = None
    '''

    print('\nStart testing')

    # always testing at the end of the code for any option
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dict, test_cindex, IBS_list, all_IBS_single, total_loss, c_lower, c_upper, c_bootstrap_cindexes = test_model(args, model_list, loss_fn, device)
    #results_dict, test_cindex, IBS_list, total_loss = _summary(args.dataset_factory.test_client, model, args.modality, loss_fn, clients = args.dataset_factory.clients)
    if args.fed_option != 2:
        print('Final Test c-index: {}'.format(test_cindex[0])) # print c-index of test run
        print(f"C-index: {test_cindex[0]} (95% CI: {c_lower[0]}-{c_upper[0]})")
        #print('Bootstrap c-indexes: {}'.format(c_bootstrap_cindexes))
        if IBS_list is not []:
            print('Final Test IBS (all clients): {}'.format(IBS_list[0])) # print IBS of test run
            print('Final IBS per client: {}'.format(all_IBS_single))
            for i in range(args.num_clients):
                wandb.log({'test_IBS_c'+ str(i) + '_single': all_IBS_single[i]})

    return val_cindex, results_dict, test_cindex, IBS_list, total_loss, c_lower, c_upper, c_bootstrap_cindexes