#----> general imports
import torch
import numpy as np
import torch.nn as nn
import pdb
import os
import pandas as pd 
import warnings
import math

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = 'src/'

def _prepare_for_experiment(args):
    r"""
    Creates experiment code which will be used for identifying the experiment later on. Uses the experiment code to make results dir.
    Prints and logs the important settings of the experiment. Loads the pathway composition dataframe and stores in args for future use.

    Args:
        - args : argparse.Namespace
    
    Returns:
        - args : argparse.Namespace

    """
    # print device
    args.device = device
    print(args.device)
    # add more info in args
    #args.split_dir = os.path.join(BASE + "splits", args.which_splits, args.study)
    args.combined_study = args.study
    args = _get_custom_exp_code(args)
    _seed_torch(args.seed)

    args.fed_option = 0    # standard case, normal federated
    if args.fed_test_options == 'centralized':
        args.fed_option = 1    # centralized case, 1 client
    elif args.fed_test_options == 'islands':
        args.fed_option = 2    # clients as islands case

    #assert os.path.isdir(args.split_dir)
    #print('Split dir:', args.split_dir)

    #---> where to store the experiment related assets
    #_create_results_dir(args)

    #---> store the settings
    settings = {'federated testing option' : args.fed_test_options,
                'federated method' : args.fed_method,
                'mu' : args.mu if args.fed_method == 'fedprox' else '-',
                'lr': args.lr if args.fed_method != 'fedopt' else '-',
                'decay_reg': args.reg,
                'lr_client': args.lr_client if args.fed_method == 'fedopt' else '-',
                'lr_server': args.lr_server if args.fed_method == 'fedopt' else '-',
                'max rounds': args.max_rounds,
                'patience' : args.patience,
                'task': args.task,
                'max_epochs': args.max_epochs,  
                'experiment': args.study,
                # 'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                "num_patches":args.num_patches,
                "dropout":args.encoder_dropout,
                "type_of_path":args.type_of_path,
                'num_clients': args.num_clients,
                'datasets_path': args.dataset_path,
                'split_path': args.split_path,
                'seed': args.seed,
                'model_params_saved_in': args.model_dir,
                'final_model_in' : args.save_model_dir,
                'rna_data_path': args.omics_dir,
                }
    
    #---> bookkeping
    _print_and_log_experiment(args, settings)

    print('----------------------------')
    print('Model:')
    print('embedding dimension for pathway and wsi: ', args.wsi_projection_dim)
    print('encoding layer 1 dim: ', args.encoding_layer_1_dim)
    print('\n')

    #---> load composition df with pathways compositions, index them
    # only used for pathway only models, not in survpath
    if args.type_of_path != 'other':
        composition_df = pd.read_csv(BASE + "datasets_csv_original/pathway_compositions/{}_comps.csv".format(args.type_of_path), index_col=0)
        composition_df.sort_index(inplace=True)
        args.composition_df = composition_df

    return args


def _print_and_log_experiment(args, settings):
    r"""
    Prints the experimental settings and stores them in a file (file experiment_param_code)
    
    Args:
        - args : argspace.Namespace
        - settings : dict 
    
    Return:
        - None
        
    """
    #with open(args.results_dir + '/experiment_{}.txt'.format(args.param_code), 'w') as f:    ## TODO: removed
        #print(settings, file=f)
    #f.close()

    print("")
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("")


def _get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.
    nicer way to read info on experiment code saved in params_code in args

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)

    """
    dataset_path = BASE + 'datasets_csv/all_survival_endpoints'
    param_code = ''

    #----> Study 
    param_code += args.study + "_"

    #----> Loss Function
    param_code += '_%s' % args.bag_loss
    param_code += '_a%s' % str(args.alpha_surv)
    
    #----> Learning Rate
    param_code += '_lr%s' % format(args.lr, '.0e')

    #----> Regularization
    # if args.reg_type == 'L1':
    #   param_code += '_%sreg%s' % (args.reg_type, format(args.reg, '.0e'))

    # if args.reg and args.reg_type == "L2":
    param_code += "_l2Weight_{}".format(args.reg)

    #param_code += '_%s' % args.which_splits.split("_")[0]

    #----> Batch Size
    param_code += '_b%s' % str(args.batch_size)

    # label col 
    param_code += "_" + args.label_col

    param_code += "_dim1_" + str(args.encoding_dim)
    # param_code += "_dim2_" + str(args.encoding_layer_2_dim)
    
    param_code += "_patches_" + str(args.num_patches)
    # param_code += "_dropout_" + str(args.encoder_dropout)

    param_code += "_wsiDim_" + str(args.wsi_projection_dim)
    param_code += "_epochs_" + str(args.max_epochs)
    param_code += "_fusion_" + str(args.fusion)
    param_code += "_modality_" + str(args.modality)
    param_code += "_pathT_" + str(args.type_of_path)

    #----> Updating
    args.param_code = param_code
    #args.dataset_path = dataset_path

    return args


def _seed_torch(seed=7):
    r"""
    Sets custom seed for torch 
    
    Args:
        - seed : Int 
    
    Returns:
        - None

    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   # Sets the seed for generating random numbers on all devices
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _create_results_dir(args):
    r"""
    Creates a dir to store results for this experiment. Adds .gitignore 
    A dir for the results with subdir esperiment specific; and a subdir for the esperiment where to store param_code

    Args:
        - args: argspace.Namespace
    
    Return:
        - None 
    
    """
    args.results_dir = os.path.join("./results", args.results_dir) # create an experiment specific subdir in the results dir. Use absolute path instead of ./results
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
        #---> add gitignore to results dir
        f = open(os.path.join(args.results_dir, ".gitignore"), "w")
        f.write("*\n")
        f.write("*/\n")
        f.write("!.gitignore")
        f.close()
    
    #---> results for this specific experiment
    args.results_dir = os.path.join(args.results_dir, 'res_test')
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)


def _get_start_end(args):
    r"""
    Which folds are we training on.
    Get evenly distributed values within start and end of args.k (from 0 to 5 for default values)

    Args:
        - args : argspace.Namespace
    
    Return:
       folds : np.array 
    
    """
    # default values for k_start and k_end -1. k = 5 in scripts
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)  # Return evenly spaced values within a given interval
    return folds


def _save_splits(split_datasets, column_keys, filename, boolean_style=False):
    r'''
    If Boolean False: Put the splits in the directory indicated by filename with two columns, train and val, plus a new index
    Then prints (what?)

    Args:
        -split_databases for train and val, called datasets in core_utils
        -column keys        ['train', 'val']
        -filename      directory for results with subdir of current split
        -Boolean
    '''

    splits = [split_datasets[i].metadata['slide_id'] for i in range(len(split_datasets))] # for both train and val, take relative ds and id for the slides
    if not boolean_style:  # here for _train_val() of core_utils
        df = pd.concat(splits, ignore_index=True, axis=1)  # Concatenate pandas objects along a particular axis (1 - column)
        df.columns = column_keys
        # create a matrix, first column train, second column val, same format as in splits_n file, with new index
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val'])

    df.to_csv(filename)
    print()


def _series_intersection(s1, s2):
    r"""
    Return insersection of two sets
    
    Args:
        - s1 : set
        - s2 : set 
    
    Returns:
        - pd.Series
    
    """
    return pd.Series(list(set(s1) & set(s2)))


def _print_network(results_dir, net):
    r"""

    Print the model in terminal and also to a text file for storage 
    
    Args:
        - results_dir : String 
        - net : PyTorch model 
    
    Returns:
        - None 
    
    """
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

    # print(net)

    #fname = "model_" + results_dir.split("/")[-1] + ".txt"
    #path = os.path.join(results_dir, fname)
    #f = open(path, "w")
    #f.write(str(net))
    #f.write("\n")
    #f.write('Total number of parameters: %d \n' % num_params)
    #f.write('Total number of trainable parameters: %d \n' % num_params_train)
    #f.close()


def _collate_omics(batch):
    r"""
    Collate function for the unimodal omics models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
  
    img = torch.ones([1,1])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    return [img, omics, label, event_time, c, clinical_data_list]


def _collate_wsi_omics(batch):
    r"""
    Collate function for the unimodal wsi and multimodal wsi + omics  models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
  
    img = torch.stack([item[0] for item in batch])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omics, label, event_time, c, clinical_data_list, mask]


def _collate_MCAT(batch):
    r"""
    Collate function MCAT (pathways version) model
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic1 : torch.Tensor 
        - omic2 : torch.Tensor 
        - omic3 : torch.Tensor 
        - omic4 : torch.Tensor 
        - omic5 : torch.Tensor 
        - omic6 : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
    
    img = torch.stack([item[0] for item in batch])

    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)


    label = torch.LongTensor([item[7].long() for item in batch])
    event_time = torch.FloatTensor([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[10])

    mask = torch.stack([item[11] for item in batch], dim=0)

    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data_list, mask]


def _collate_survpath(batch):
    r"""
    Collate function for survpath
    Get info as img from item[0], omic_data_list ([1]), label ([2]), event_time ([3]), c censored ([4]), clinical_data_list ([5]) from all items of batch.
    Put them in a list and return it
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic_data_list : List
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
    is_wsi = True
    for item in batch:
        if item[0] is None:
            is_wsi = False
            break
    if is_wsi:
        img = torch.stack([item[0] for item in batch])
    else:
        img = None

    omic_data_list = []
    for item in batch:
        if item[1] is None:
            omic_data_list = None
            break
        else:
            omic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    if not is_wsi:
        mask = None
    else:
        mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omic_data_list, label, event_time, c, clinical_data_list, mask]


def _make_weights_for_balanced_classes_split(dataset):
    r"""
    Returns the weights for each class. The class will be sampled proportionally.
    Code is adapted to manage clients for wich some classes don't have any patients.
    If at least half of the classes are empty it throws a warning

    Args: 
        - dataset : SurvivalDataset
    
    Returns:
        - final_weights : torch.DoubleTensor 
    
    """
    '''N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                   
        weight[idx] = weight_per_class[y]   

    final_weights = torch.DoubleTensor(weight)

    return final_weights'''

    N = float(len(dataset))
    n_classes = len(dataset.slide_cls_ids)

    weight_per_class = []
    for c in range(n_classes):
        cnt = len(dataset.slide_cls_ids[c])
        if cnt > 0:
            weight_per_class.append(N / cnt)
        else:
            weight_per_class.append(0.0)
            warnings.warn(f"Class {c} has 0 samples on this client — set weight to 0.")
    
    num_zeroes = weight_per_class.count(0.0)
    if num_zeroes > math.floor(n_classes/2):
        warnings.warn(f'half of the classes are empty, please don\'t train the model and fix the problem')

    weight = [0.0] * int(N)
    for idx in range(int(N)):
        y = int(dataset.getlabel(idx))   # ensure y is an int
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)


def _get_split_loader(args, split_dataset, training = False, testing = False, weighted = False, batch_size=1):
    r"""
    Take a dataset and make a dataloader from it using a custom (from modality) collate function. 

    Args:
        - args : argspace.Namespace
        - split_dataset : SurvivalDataset
        - training : Boolean
        - testing : Boolean
        - weighted : Boolean 
        - batch_size : Int 
    
    Returns:
        - loader : Pytorch Dataloader 
    
    """
    #print('entered the get_split_loader function')
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    # get loader collate_fn according to type of modality selected, get important info from all items of batch
    if args.modality in ["omics", "snn", "mlp_per_path"]:
        collate_fn = _collate_omics
    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        collate_fn = _collate_wsi_omics
    elif args.modality in ["coattn", "coattn_motcat"]:  
        collate_fn = _collate_MCAT
    elif args.modality == "survpath":
        collate_fn = _collate_survpath 
    else:
        raise NotImplementedError

    # get DataLoaders for train or validation (or testing), divide the classes balanced among the splits
    if args.loader_sampler == 1:
        if not testing:
            if training:
                if weighted:
                    weights = _make_weights_for_balanced_classes_split(split_dataset)
                    loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, drop_last=False, **kwargs)	
                else:
                    loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)

        else:  # only randomizes 10% of the data, not useful and does not work
            ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, drop_last=False, **kwargs )
    
    else:
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = None, collate_fn = collate_fn, drop_last=False, **kwargs)

    return loader



def save_model(args, model_list):
    r'''
    save the model parameters in the directory assigned

    '''
    if args.fed_option == 2: # islands
        # remove '.pth' from directory
        try:
            dir = args.save_model_dir.replace('.pth', '')
            for cid, model in enumerate(model_list):
                #model.load_state_dict(torch.load(dir + '_c' + str(cid) + '.pth'))
                torch.save(model.state_dict(), dir + '_c' + str(cid) + '.pth')
        except Exception as e:
            print('something went wrong with the saving: ', e)
        
        return
     
    try:
        #model_list[0].load_state_dict(torch.load(args.save_model_dir))
        torch.save(model_list[0].state_dict(), args.save_model_dir)
    except Exception as e:
        print('something went wrong while saving the model\'s parameters: ', e)



from copy import deepcopy

def to_wandb_format(l, name) -> dict:
    r"""
    Unpack list values in the dictionary, as wandb can't plot list values.
    Example:
        Input: [99, 88, 77], 'name'
        Output: {"name_0": 99, "name_1": 88, "name_2": 77}
    Works with any type in input
    """
    
    if isinstance(l, (int, float, str)):
        return {name: l}

    d = {}
    if isinstance(l, list):
        d.update({
            f'{name}_c{id}': el for id, el in enumerate(l)
        })
        return d
    
    if isinstance(l, dict):
        for key, val in l.items():
            if isinstance(val, list):
                d.update({
                    f'{name}_c{i}': v for i, v in enumerate(val)
                })
    
    return d


def save_metrics(args, test_cindex, test_IBS_list, total_loss, c_lower, c_upper, c_bootstrap_cindexes):
    r'''
    Save the metrics on wandb
    '''
    wandb.log({
        "test_cindex_all": test_cindex,
        "tes_c_lower_all": c_lower,
        "test_c_upper_all": c_upper,
        "test_c_bootstrap_cindexes_all": c_bootstrap_cindexes,
        "test_loss_all": total_loss,
        "test_IBS_list": test_IBS_list
    })
    
    wandb.log(to_wandb_format(test_cindex, 'test_cindex')) 
    wandb.log(to_wandb_format(c_lower, 'test_c_lower'))
    wandb.log(to_wandb_format(c_upper, 'test_c_upper'))
    wandb.log(to_wandb_format(c_bootstrap_cindexes, 'test_c_bootstrap_cindexes'))
    wandb.log(to_wandb_format(total_loss, 'test_loss'))
    wandb.log(to_wandb_format(test_IBS_list, 'test_IBS'))
    
    for i in range(len(test_cindex)):
        if args.fed_option == 2:
            wandb.log({
                "test_cindex_c" + str(i): test_cindex[i],
                "test_c_lower_c" + str(i): c_lower[i],
                "test_c_upper_c" + str(i): c_upper[i],
                "test_c_bootstrap_cindexes_c" + str(i): c_bootstrap_cindexes[i],
                "test_loss_c" + str(i): total_loss[i],
                "test_IBS_c" + str(i): test_IBS_list[i]
            })
        else:
            wandb.log({
                "test_cindex": test_cindex[i],
                "test_c_lower": c_lower[i],
                "test_c_upper": c_upper[i],
                "test_c_bootstrap_cindexes": c_bootstrap_cindexes[i],
                "test_loss": total_loss[i],
                "test_IBS_c" + str(i): test_IBS_list[i]
            })
    

def print_results(args, results, test_cindex, test_IBS_list, total_loss, c_lower, c_upper, c_bootstrap_cindexes):
    r'''
    Print the metrics
    '''
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-detect terminal width
    pd.set_option('display.max_colwidth', None)  # Show full column content

    print(f'\n\n###########\n final_df: \n')
    print(f'test_cindex: {test_cindex}\n')
    print(f'95% CI lower bounds: {c_lower}')
    print(f'95% CI upper bounds: {c_upper}\n')
    print(f'Bootstrap c-indexes: {c_bootstrap_cindexes}\n')
    print(f'test_loss: {total_loss}\n')
    for cid, ibs in enumerate(test_IBS_list):
        print(f"IBS for Client {cid}: {ibs}")
        wandb.log({
            "test_IBS_c" + str(cid): ibs
        })

    print('\n### results ####')
    print(results)
    print('\n')