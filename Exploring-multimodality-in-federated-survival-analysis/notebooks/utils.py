import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import statistics
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from sklearn.preprocessing import MinMaxScaler

from src.utils.general_utils import _series_intersection
from src.datasets.dataset_survival import SurvivalDataset

from src.utils.loss_func import NLLSurvLoss
from src.models.model_SurvPath import SurvPath
from src.utils.core_utils import _unpack_data, _calculate_risk, _update_arrays

from src.zoorvival.data import load_tcga_data
from src.zoorvival.nn.training import as_torch_dataset



signature = 'other'
censorship_var = 'censorship_os'
label_col = 'survival_months_os'


def get_scaler_datasets(client, set_i, omics_data, db, study):
    r'''
    Generates a scaler from the whole dataset and uses it to normalize the datasets of the clients, test and val set.
    return for each client /test and val included) a dataset, saved into client.dataset.

    '''

    # create a scaler for the clients and test val sets considering the whole dataset (omics data) not federated
    scaler = get_scaler_from_df(omics_data)

    # for each client get train dataset and scaler, and the validation set dataset
    if set_i == 'train':
        dataset, scaler = get_split_from_df(client=client, scaler=scaler, set_i=set_i, db=db, study=study, is_test=False, is_val=False)
        client['dataset'] = dataset
        client['scaler'] = scaler

    # for the test set
    else:
        dataset_t = get_split_from_df(client=client, scaler=scaler, set_i=set_i, db=db, study=study, is_test=True)
        client['dataset'] = dataset_t

    print("Created train, test and val datasets")
    return client


def get_scaler_from_df(omics_data):
    r"""
    Obtain a scaler from the whole dataset using omics data. We are using the case_ids, 
    so it would be better to have a second version of the dataset that contains different values in their place
    
    """

    scaler = {}

    # contains modalities for omics data
    
    filtered_df = omics_data  #raw_data_df[mask]   # only mask
    filtered_df = filtered_df[~filtered_df.index.duplicated()] # drop duplicate case_ids
    filtered_df.reset_index(inplace=True, drop=True)

    # flatten the df into 1D array (make it a column vector)
    flat_df = filtered_df.values.flatten().reshape(-1, 1)    #df_for_norm.values.flatten().reshape(-1, 1)
    
    # get scaler
    scaler_for_data = _get_scaler(flat_df)

    # store scaler
    scaler = scaler_for_data

    return scaler


def get_split_from_df(client, scaler, set_i, db, study, is_test=False, is_val=False, valid_cols=None):
    r"""
    Apply scaler on datasets -- Standardize features by removing the mean and scaling to unit variance.
    Create new dataset formatted SurvivalDataset for the data of the client, returns it
    """

    split = client['ids'][set_i]
    split = split.dropna().reset_index(drop=True)
    print("Total number of patients for the client in the set {}: {}".format(set_i, len(split))) 

    mask = client['labels']['case_id'].isin(split.tolist())
    df_metadata_slide = client['labels'].loc[mask, :].reset_index(drop=True)   # take the corresponding samples from label data
    
    # select the rna, meth (drugs?), mut (mutations in rna), cnv (type of deviation in rna) data for this split
    omics_data_for_split = {}
     
    raw_data_df = client['omic']
    mask = raw_data_df.index.isin(split.tolist())   # new mask between split index and modality data
    
    filtered_df = raw_data_df[mask]    # raw_data_df[mask]   # only mask
    filtered_df = filtered_df[~filtered_df.index.duplicated()] # drop duplicate case_ids
    filtered_df["temp_index"] = filtered_df.index
    filtered_df.reset_index(inplace=True, drop=True)
    #print('filtered_df omic: ', filtered_df.head(3))

    clinical_data_mask = client['clinical'].case_id.isin(split.tolist())  # same with clinical data, new mask
    clinical_data_for_split = client['clinical'][clinical_data_mask]
    clinical_data_for_split = clinical_data_for_split.set_index("case_id")   # new index
    clinical_data_for_split = clinical_data_for_split.replace(np.nan, "N/A")  # keep na data

    # from metadata (label data) and clinical data drop any cases that are not in filtered_df
    # keep patients in label data that are also in filtered df (patients that don't have omics data)

    mask = [True if item in list(filtered_df["temp_index"]) else False for item in clinical_data_for_split.index]
    clinical_data_for_split = clinical_data_for_split[mask]
    clinical_data_for_split = clinical_data_for_split[~clinical_data_for_split.index.duplicated(keep='first')]

    # normalize your df 
    filtered_normed_df = None
    if is_test:   ## apply scaler to ds
        
        # store the case_ids -> create a new df without case_ids
        case_ids = filtered_df["temp_index"]
        df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

        # store original num_patients and num_feats 
        num_patients = df_for_norm.shape[0]
        num_feats = df_for_norm.shape[1]
        columns = {}
        for i in range(num_feats):
            columns[i] = df_for_norm.columns[i]
        
        # flatten the df into 1D array (make it a column vector)
        flat_df = np.expand_dims(df_for_norm.values.flatten(), 1)

        # get scaler for the current modality
        scaler_for_data = scaler
        #scaler_for_data = scaler

        # normalize 
        normed_flat_df = _apply_scaler(data = flat_df, scaler = scaler_for_data)

        # change 1D to 2D
        filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

        # add in case_ids
        filtered_normed_df["temp_index"] = case_ids
        filtered_normed_df.rename(columns=columns, inplace=True)
        #print('filtered_normed_df: ', filtered_normed_df.head(3))

    else:
        
        # store the case_ids -> create a new df without case_ids
        case_ids = filtered_df["temp_index"]
        df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

        # store original num_patients and num_feats 
        num_patients = df_for_norm.shape[0]
        num_feats = df_for_norm.shape[1]
        columns = {}
        for i in range(num_feats):
            columns[i] = df_for_norm.columns[i]
        
        # flatten the df into 1D array (make it a column vector)
        flat_df = df_for_norm.values.flatten().reshape(-1, 1)
        
        # get scaler
        scaler_for_data = scaler

        # normalize 
        normed_flat_df = _apply_scaler(data = flat_df, scaler = scaler_for_data)

        # change 1D to 2D
        filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

        # add in case_ids
        filtered_normed_df["temp_index"] = case_ids
        filtered_normed_df.rename(columns=columns, inplace=True)

        # store scaler
        scaler = scaler_for_data
        
    omics_data_for_split = filtered_normed_df
    #print('omic data: ', omics_data_for_split.head(3))

    if not is_test:
        sample = True
    else:
        sample = False

    # get zoorvival project
    #db = load_tcga_data(study)

    # modify the omic data so that it's in the format of the original code
    omics_data = {}
    omics_data['rna'] = omics_data_for_split
        
    split_dataset = SurvivalDataset(
        client_id = set_i,
        study_name = study,
        modality = 'survpath',
        patient_dict = client['patient_dict'],
        metadata = df_metadata_slide,    # get label data
        omics_data_dict = omics_data,  # get normalized omics dat
        data_wsi = db,    # get wsi images
        num_classes = 4,
        label_col = label_col,
        censorship_var = censorship_var,
        valid_cols = valid_cols,
        is_test = is_test == False,   ## ????
        clinical_data = clinical_data_for_split,  # get clinicla data
        num_patches = 4096,
        omic_names = client['omic_names'],
        sample = sample
        )

    #print('dataset: ', split_dataset)
    if is_test:
        return split_dataset    # returns the SurvivalDataset
    else:
        return split_dataset, scaler



def _apply_scaler(data, scaler):
        r"""
        Given the datatype and a predefined scaler, apply it to the data 
        
        """
        
        # find out which values are missing
        zero_mask = data == 0

        # transform data
        transformed = scaler.transform(data)
        data = transformed

        # rna -> put back in the zeros 
        data[zero_mask] = 0.
        
        return data

def _get_scaler(data):
        r"""
        Define the scaler for training dataset. Use the same scaler for validation set
        MinMaxSCaler: Transform features by scaling each feature to a given range.
        This estimator scales and translates each feature individually such that it is in the given range on the training set
        
        """
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        return scaler



def _get_patient_dict(client=None):
    r"""
    For every patient store the respective slide ids (key for wsi(s)) in self.patient_df where patient_df = {case_id of patient: [list of slide_id of patient]; ...}
    """

    patient_dict = {}
    temp_label_data = client['labels'].set_index('case_id')  # create an index for the df using the column 'case_id'
    for patient in client['patients_df']['case_id']:
        slide_ids = temp_label_data.loc[patient, 'slide_id']  # Access a group of rows and columns by label(s) or a boolean array
        if isinstance(slide_ids, str):
            slide_ids = np.array(slide_ids).reshape(-1)
        else:
            slide_ids = slide_ids.values
        patient_dict.update({patient:slide_ids})
    client['patient_dict'] = patient_dict
    client['labels'] = client['patients_df']
    client['labels'].reset_index(drop=True, inplace=True)

    return client



def _get_split_loader(split_dataset, device, batch_size=1):
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
    kwargs = {'num_workers': 4} if device == "cuda" else {}
    # get loader collate_fn according to type of modality selected, get important info from all items of batch
    
    collate_fn = _collate_survpath 

    # get DataLoaders for train or validation (or testing), divide the classes balanced among the splits
    loader = DataLoader(split_dataset, batch_size=batch_size, sampler = None, collate_fn = collate_fn, drop_last=False, **kwargs)

    return loader


def _collate_survpath(batch):
    r"""
    Collate function for survpath
    Get info as img from item[0], omic_data_list ([1]), label ([2]), event_time ([3]), c censored ([4]), clinical_data_list ([5]) from all items of batch.
    Put them in a list and return it
    
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



def _init_loss_function(alpha_surv):
    print('\nInit loss function...', end=' ')
    loss_fn = NLLSurvLoss(alpha=alpha_surv)  # loss function taken from other repository, little documentation in this one
    print('Done!')
    return loss_fn




'''
IG STUFF

import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
from captum.attr._utils.visualization import (
    VisualizationDataRecord,
    visualize_image_attr,
)
import matplotlib.pyplot as plt
#import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float32)

class MultimodalIntegratedGradients:
    """Handles Integrated Gradients for multimodal survival model"""
    
    def __init__(self, model, omic_sizes):
        self.model = model
        self.omic_sizes = omic_sizes
        self.model.eval()
        
        # Create IG instances for each modality
        self.ig_wsi = IntegratedGradients(self._forward_wsi, omic_sizes)
        self.ig_omic1 = IntegratedGradients(self._forward_omic1, omic_sizes)
        self.ig_omic2 = IntegratedGradients(self._forward_omic2, omic_sizes)
        self.ig_omic3 = IntegratedGradients(self._forward_omic3, omic_sizes)
        self.ig_omic4 = IntegratedGradients(self._forward_omic4, omic_sizes)
        
        # For layer attributions (intermediate representations)
        #self.layer_ig = LayerIntegratedGradients(model, model.fusion[0])
        
    def _forward_wsi(self, wsi):
        """Forward pass for WSI-only attribution"""
        # Create dummy omic inputs (zeros)
        device = wsi.device
        batch_size = 1
        omic_dims = self.omic_sizes
        dummy_omics = [
            torch.zeros(batch_size, dim, device=device) for dim in omic_dims ]
        return self.model.captum( *dummy_omics, wsi)
    
    def _forward_omic1(self, omic1):
        """Forward pass for omic1-only attribution"""
        device = omic1.device
        batch_size = 1
        dummy_wsi = torch.zeros(batch_size, 64, 1536, device=device)
        dummy_omic2 = torch.zeros(batch_size, self.omic_sizes[1], device=device)
        dummy_omic3 = torch.zeros(batch_size, self.omic_sizes[2], device=device)
        dummy_omic4 = torch.zeros(batch_size, self.omic_sizes[3], device=device)
        return self.model.captum( omic1, dummy_omic2, dummy_omic3, dummy_omic4, dummy_wsi)
    
    # Similar methods for omic2, omic3, omic4...
    def _forward_omic2(self, omic2):
        device = omic2.device
        batch_size = 1
        dummy_wsi = torch.zeros(batch_size, 64, 1536, device=device)
        dummy_omic1 = torch.zeros(batch_size, self.omic_sizes[0], device=device)
        dummy_omic3 = torch.zeros(batch_size, self.omic_sizes[2], device=device)
        dummy_omic4 = torch.zeros(batch_size, self.omic_sizes[3], device=device)
        return self.model.captum(dummy_omic1, omic2, dummy_omic3, dummy_omic4, dummy_wsi )
    
    def _forward_omic3(self, omic3):
        device = omic3.device
        batch_size = 1
        dummy_wsi = torch.zeros(batch_size, 64, 1536, device=device)
        dummy_omic1 = torch.zeros(batch_size, self.omic_sizes[0], device=device)
        dummy_omic2 = torch.zeros(batch_size, self.omic_sizes[1], device=device)
        dummy_omic4 = torch.zeros(batch_size, self.omic_sizes[3], device=device)
        return self.model.captum(dummy_omic1, dummy_omic2, omic3, dummy_omic4, dummy_wsi)
    
    def _forward_omic4(self, omic4):
        device = omic4.device
        batch_size = 1
        dummy_wsi = torch.zeros(batch_size, 64, 1536, device=device)
        dummy_omic1 = torch.zeros(batch_size, self.omic_sizes[0], device=device)
        dummy_omic2 = torch.zeros(batch_size, self.omic_sizes[1], device=device)
        dummy_omic3 = torch.zeros(batch_size, self.omic_sizes[2], device=device)
        return self.model.captum( dummy_omic1, dummy_omic2, dummy_omic3, omic4, dummy_wsi)
    
    def attribute(
        self,
        inputs: Dict[str, torch.Tensor],
        target: Optional[int] = None,
        n_steps: int = 50,
        method: str = 'gausslegendre',
        return_convergence_delta: bool = False
    ) -> Dict[str, Union[torch.Tensor, Tuple]]:
        """
        Compute Integrated Gradients attributions for all modalities
        
        Args:
            inputs: Dictionary with keys 'wsi', 'omic1', 'omic2', 'omic3', 'omic4'
            target: Target class index (for classification) or None for regression
            n_steps: Number of steps for IG approximation
            method: Integration method ('riemann_trapezoid' or 'gausslegendre')
        """
        attributions = {}
        deltas = {}
        
        # Create baselines (zeros for all modalities)
        baselines = {
            'wsi': torch.zeros_like(inputs['wsi']),
            'omic1': torch.zeros_like(inputs['omic1']),
            'omic2': torch.zeros_like(inputs['omic2']),
            'omic3': torch.zeros_like(inputs['omic3']),
            'omic4': torch.zeros_like(inputs['omic4'])
        }
        
        # WSI attribution
        attr_wsi, delta_wsi = self.ig_wsi.attribute(
            inputs['wsi'],
            baselines=baselines['wsi'],
            target=target,
            n_steps=n_steps,
            method=method,
            return_convergence_delta=True
        )
        attributions['wsi'] = attr_wsi
        deltas['wsi'] = delta_wsi
        
        # Omic attributions
        omic_igs = [self.ig_omic1, self.ig_omic2, self.ig_omic3, self.ig_omic4]
        for i, (omic_name, ig) in enumerate(zip(['omic1', 'omic2', 'omic3', 'omic4'], omic_igs)):
            attr_omic, delta_omic = ig.attribute(
                inputs[omic_name],
                baselines=baselines[omic_name],
                target=target,
                n_steps=n_steps,
                method=method,
                return_convergence_delta=True
            )
            attributions[omic_name] = attr_omic
            deltas[omic_name] = delta_omic
        
        if return_convergence_delta:
            return attributions, deltas
        return attributions
    
    def attribute_layer(
        self,
        inputs: Dict[str, torch.Tensor],
        target: Optional[int] = None,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute attributions for intermediate layer (fusion layer)
        """
        raise NotImplementedError("Layer attribution not configured for this model")



class AttributionVisualizer:
    """Visualization tools for attributions"""
    
    @staticmethod
    def visualize_wsi_attributions(
        wsi_image: np.ndarray,
        attributions: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualize WSI attributions as heatmap overlay
        """
        # Convert attributions to numpy
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.cpu().detach().numpy()
        
        # Handle multi-channel attributions (average if needed)
        if len(attributions.shape) == 3:
            attributions = np.mean(attributions, axis=0)
        
        # Normalize
        attr_norm = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(wsi_image)
        ax1.set_title('Original WSI')
        ax1.axis('off')
        
        # Attribution heatmap
        im = ax2.imshow(attr_norm, cmap='viridis')
        ax2.set_title('Attribution Heatmap')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Overlay
        ax3.imshow(wsi_image, alpha=0.7)
        ax3.imshow(attr_norm, cmap='jet', alpha=0.3)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def visualize_omic_attributions(
        gene_names: List[str],
        attributions: torch.Tensor,
        top_k: int = 20,
        title: str = "Top Gene Attributions",
        save_path: Optional[str] = None
    ):
        """
        Visualize top contributing genes
        """
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.cpu().detach().numpy()
        
        # Get absolute values for sorting
        abs_attr = np.abs(attributions)
        top_indices = np.argsort(abs_attr)[-top_k:][::-1]
        
        # Prepare data
        top_genes = [gene_names[i] for i in top_indices]
        top_attr = attributions[top_indices]
        colors = ['red' if x > 0 else 'blue' for x in top_attr]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(top_k), top_attr, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_genes)
        ax.set_xlabel('Attribution Score')
        ax.set_title(title)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_attr)):
            ax.text(val, i, f'{val:.3f}', 
                   va='center', ha='left' if val > 0 else 'right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def visualize_time_dependent_attributions(
        time_points: List[float],
        attributions_over_time: Dict[float, Dict[str, torch.Tensor]],
        modality: str,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
        save_path: Optional[str] = None
    ):
        """
        Visualize how attributions change over time for a specific modality
        """
        fig, axes = plt.subplots(1, len(time_points), figsize=(15, 4))
        
        for i, (t, attrs) in enumerate(attributions_over_time.items()):
            attr = attrs[modality]
            if isinstance(attr, torch.Tensor):
                attr = attr.cpu().detach().numpy()
            
            if len(attr.shape) > 1:
                attr = attr.flatten()
            
            # Get top features for this time point
            top_indices = np.argsort(np.abs(attr))[-top_k:]
            
            if feature_names:
                labels = [feature_names[j] for j in top_indices]
            else:
                labels = [f'Feature {j}' for j in top_indices]
            
            axes[i].barh(range(top_k), attr[top_indices])
            axes[i].set_yticks(range(top_k))
            axes[i].set_yticklabels(labels)
            axes[i].set_title(f't = {t}')
            axes[i].set_xlabel('Attribution')
        
        plt.suptitle(f'{modality} Attributions Over Time')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_attribution_report(
        attributions: Dict[str, torch.Tensor],
        gene_names: Optional[Dict[str, List[str]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive attribution report
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots for each modality
        n_modalities = len(attributions)
        n_cols = 3
        n_rows = (n_modalities + n_cols - 1) // n_cols
        
        for i, (modality, attr) in enumerate(attributions.items()):
            if modality == 'wsi':
                continue  # Handle WSI separately
            
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            
            if isinstance(attr, torch.Tensor):
                attr = attr.cpu().detach().numpy()
            
            if len(attr.shape) > 1:
                attr = attr.flatten()
            
            # Get top 15 features
            top_k = min(15, len(attr))
            top_indices = np.argsort(np.abs(attr))[-top_k:][::-1]
            
            if gene_names and modality in gene_names:
                labels = [gene_names[modality][j] for j in top_indices]
            else:
                labels = [f'{modality}_{j}' for j in top_indices]
            
            colors = ['red' if x > 0 else 'blue' for x in attr[top_indices]]
            ax.barh(range(top_k), attr[top_indices], color=colors)
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f'{modality} Top Features')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle('Multimodal Attribution Analysis Report')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 5. Initialize the visualizer
#visualizer = AttributionVisualizer()


# 6. NOW CALL THE VISUALIZER METHODS
visualizer.visualize_wsi_attributions(
    wsi_image=np.random.rand(256, 256, 3),  # Replace with actual WSI image
    attributions=attributions['wsi']
)
visualizer.visualize_omic_attributions(
    gene_names=[f'Gene_{i}' for i in range(test_client['omic_sizes'][0])],  # Replace with actual gene names
    attributions=attributions['omic1'],
    title="Omic1 Attributions"
)


torch.set_default_dtype(torch.float32)
omic_sizes = split_test[0]['omic_sizes']
# 3. Prepare your input data
all_mean_importance = []

for i in range(num_splits):
    test_client = split_test[i]
    model = algo_models[current_algo][i]
    ig = ig_all[i]

    split_importance = []

    print(f"-------   Split {i}: -------")

    with torch.no_grad():
        for data in test_client['loader']:
            
            data_WSI, _, y_disc, event_time, censor, data_omics, _, _ = _unpack_data('survpath', device, data)

            batch_size = 1
            inputs = {
                'wsi': data_WSI.to(device) if data_WSI is not None else torch.zeros(batch_size, 64, 1536, device=device),
                'omic1': data_omics[0].type(torch.FloatTensor).to(device).unsqueeze(0),
                'omic2': data_omics[1].type(torch.FloatTensor).to(device).unsqueeze(0),
                'omic3': data_omics[2].type(torch.FloatTensor).to(device).unsqueeze(0),
                'omic4': data_omics[3].type(torch.FloatTensor).to(device).unsqueeze(0)
            }
            #inputs = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
            #baselines = {k: v.to(dtype=torch.float32) for k, v in baselines.items()}

            # 4. Compute attributions
            print("Computing Integrated Gradients...")
            attributions = ig.attribute(
                inputs=inputs,
                n_steps=1,
                method='gausslegendre'
            )

            gradient = GradientBasedModalityImportance(model, device=device, omic_sizes=omic_sizes)
            importance = gradient.compute_gradient_importance(inputs)
            print("Modality Importance (Raw), single patient:", importance['raw'])
            print("Modality Importance (Relative), single patient:", importance['relative'])
            split_importance.append(importance)

    split_mean_importance = {
        'raw': {
            'WSI': np.mean([imp['raw']['WSI'] for imp in split_importance]),
            'CNV': np.mean([imp['raw']['CNV'] for imp in split_importance]),
            'DNAm': np.mean([imp['raw']['DNAm'] for imp in split_importance]),
            'miRNA': np.mean([imp['raw']['miRNA'] for imp in split_importance]),
            'mRNA': np.mean([imp['raw']['mRNA'] for imp in split_importance])
        },
        'relative': {
            'WSI': np.mean([imp['relative']['WSI'] for imp in split_importance]),
            'CNV': np.mean([imp['relative']['CNV'] for imp in split_importance]),
            'DNAm': np.mean([imp['relative']['DNAm'] for imp in split_importance]),
            'miRNA': np.mean([imp['relative']['miRNA'] for imp in split_importance]),
            'mRNA': np.mean([imp['relative']['mRNA'] for imp in split_importance])
        }
    }
    print(f"-------   Split {i} Mean Importance: -------")
    print("Modality Importance (Raw), single patient:", split_mean_importance['raw'])
    print("Modality Importance (Relative), single patient:", split_mean_importance['relative'])
    print('\n')
    all_mean_importance.append(split_mean_importance)


def plot_importances(modalities, raw_importance, relative_importance, n_modalities, n_splits):

    # Calculate aggregate statistics for relative importance
    means = np.mean(relative_importance, axis=1)
    stds = np.std(relative_importance, axis=1)
    medians = np.median(relative_importance, axis=1)
    q1 = np.percentile(relative_importance, 25, axis=1)
    q3 = np.percentile(relative_importance, 75, axis=1)
    mins = np.min(relative_importance, axis=1)
    maxs = np.max(relative_importance, axis=1)

    # Colors for modalities (using a professional color palette)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B8F5E']


    # Create the comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])

    # Main title
    fig.suptitle('Multimodal Feature Importance Analysis - Integrated Gradients\n(Aggregated Across 5 Cross-Validation Splits)', 
                fontsize=16, fontweight='bold', y=0.98)

    # ============================================================================
    # 1. Main bar plot with individual split points (top left)
    # ============================================================================
    ax1 = plt.subplot(gs[0, 0])
    x_pos = np.arange(n_modalities)
    bar_width = 0.7

    # Plot bars with error bars
    bars = ax1.bar(x_pos, means, bar_width, yerr=stds, capsize=8, 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                error_kw={'elinewidth': 2, 'capthick': 2})

    # Add individual split points with slight jitter
    for i in range(n_modalities):
        split_values = relative_importance[i]
        x_jittered = np.random.normal(x_pos[i], 0.05, n_splits)
        scatter = ax1.scatter(x_jittered, split_values, color='black', s=80, 
                            alpha=0.7, zorder=5, marker='o', edgecolors='white', linewidth=1)

    # Customize bar plot
    ax1.set_ylabel('Relative Importance (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Modalities', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Importance with Standard Deviation', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(modalities, fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(maxs) * 1.15)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(i, mean + std + 1.5, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add legend
    ax1.legend([scatter], ['Individual splits'], loc='upper right', fontsize=10)

    # ============================================================================
    # 2. Box plot with mean markers (top middle)
    # ============================================================================
    ax2 = plt.subplot(gs[0, 1])
    box_data = [relative_importance[i] for i in range(n_modalities)]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=modalities,
                    showmeans=True, meanline=True, 
                    meanprops={'color': 'red', 'linewidth': 2, 'label': 'Mean'},
                    medianprops={'color': 'blue', 'linewidth': 2, 'label': 'Median'})

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Customize boxplot
    ax2.set_ylabel('Relative Importance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution Across Splits\n(Box: 25-75%, Whisker: Min-Max)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45, labelsize=10)

    # Add legend
    ax2.plot([], [], 'r-', label='Mean', linewidth=2)
    ax2.plot([], [], 'b-', label='Median', linewidth=2)
    ax2.legend(loc='upper right', fontsize=10)

    # ============================================================================
    # 3. Heatmap showing split-wise importance (top right)
    # ============================================================================
    ax3 = plt.subplot(gs[0, 2])
    im = ax3.imshow(relative_importance.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)

    # Add text annotations
    for i in range(n_modalities):
        for j in range(n_splits):
            text = ax3.text(i, j, f'{relative_importance[i, j]:.1f}',
                        ha='center', va='center', fontsize=9,
                        color='black' if relative_importance[i, j] < 50 else 'white')

    # Customize heatmap
    ax3.set_xticks(np.arange(n_modalities))
    ax3.set_yticks(np.arange(n_splits))
    ax3.set_xticklabels(modalities, fontsize=10, fontweight='bold')
    ax3.set_yticklabels([f'Split {j+1}' for j in range(n_splits)], fontsize=10)
    ax3.set_title('Split-wise Importance Heatmap', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Modalities', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CV Splits', fontsize=11, fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax3, label='Relative Importance (%)', shrink=0.8)

    # ============================================================================
    # 4. Raw vs Relative comparison (bottom left)
    # ============================================================================
    ax4 = plt.subplot(gs[1, 0])
    x = np.arange(n_splits)
    width = 0.15

    for i, modality in enumerate(modalities):
        offset = (i - 2) * width
        bars = ax4.bar(x + offset, raw_importance[i], width, label=modality,
                    color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

    ax4.set_xlabel('Cross-Validation Splits', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Raw Importance Score', fontsize=11, fontweight='bold')
    ax4.set_title('Raw Importance Across Splits', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Split {i+1}' for i in range(n_splits)])
    ax4.legend(loc='upper right', ncol=2, fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # ============================================================================
    # 5. Line plot showing trends (bottom middle)
    # ============================================================================
    ax5 = plt.subplot(gs[1, 1])
    for i, modality in enumerate(modalities):
        ax5.plot(range(1, n_splits+1), relative_importance[i], 
                marker='o', linewidth=2, markersize=8, label=modality,
                color=colors[i], markerfacecolor='white', markeredgewidth=2)

    ax5.set_xlabel('Cross-Validation Splits', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Relative Importance (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Relative Importance Trends Across Splits', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(1, n_splits+1))
    ax5.legend(loc='upper right', ncol=2, fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ============================================================================
    # 6. Summary statistics panel (bottom right)
    # ============================================================================
    ax6 = plt.subplot(gs[1, 2])
    ax6.axis('off')

    # Create summary text
    summary_text = "SUMMARY STATISTICS\n"
    summary_text += "="*45 + "\n"
    summary_text += f"{'Modality':<8} {'Mean±Std':>12} {'Median':>8} {'Range':>14}\n"
    summary_text += "-"*45 + "\n"

    for i, modality in enumerate(modalities):
        summary_text += f"{modality:<8} {means[i]:>6.1f}±{stds[i]:>4.1f} {medians[i]:>7.1f} "
        summary_text += f"{mins[i]:>4.1f}-{maxs[i]:>4.1f}\n"

    summary_text += "="*45 + "\n\n"
    summary_text += "CONFIDENCE INTERVALS (95%)\n"
    summary_text += "="*45 + "\n"

    # Calculate 95% confidence intervals
    for i, modality in enumerate(modalities):
        ci = stats.t.interval(0.95, n_splits-1, loc=means[i], scale=stds[i]/np.sqrt(n_splits))
        summary_text += f"{modality:<8}: [{ci[0]:>5.1f}, {ci[1]:>5.1f}]\n"

    summary_text += "="*45 + "\n\n"
    summary_text += f"KEY FINDINGS:\n"
    summary_text += f"• Most important: {modalities[np.argmax(means)]} ({means.max():.1f}%)\n"
    summary_text += f"• Least important: {modalities[np.argmin(means)]} ({means.min():.1f}%)\n"
    summary_text += f"• Most stable: {modalities[np.argmin(stds)]} (σ={stds.min():.2f})\n"
    summary_text += f"• Most variable: {modalities[np.argmax(stds)]} (σ={stds.max():.2f})"

    # Add text with background
    ax6.text(0, 1, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                    edgecolor='#dee2e6', alpha=0.9))

    # Add overall figure annotations
    plt.figtext(0.5, 0.01, 
                'Note: All values represent importance scores from Integrated Gradients. '
                'Error bars show standard deviation across 5 CV splits.\n'
                '95% confidence intervals calculated using t-distribution with 4 degrees of freedom.',
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='#e9ecef', alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.06, hspace=0.3, wspace=0.3)
    plt.show()

    # Print summary statistics to console
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR ALL MODALITIES")
    print("="*60)
    print(f"{'Modality':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    print("-"*60)
    for i, modality in enumerate(modalities):
        print(f"{modality:<10} {means[i]:>8.2f} {stds[i]:>8.2f} {mins[i]:>8.2f} {maxs[i]:>8.2f} {(maxs[i]-mins[i]):>8.2f}")
    print("="*60)


    class GradientBasedModalityImportance:
    def __init__(self, model, device='cuda', omic_sizes=None):
        self.model = model
        self.model.eval()
        self.device = device
        self.omic_sizes = omic_sizes
    
    def compute_gradient_importance(self, inputs):
        """
        Compute modality importance based on gradient magnitudes
        Handles missing modalities gracefully
        """
        # Check which modalities are present
        available_modalities = []
        for name in ['wsi', 'omic1', 'omic2', 'omic3', 'omic4']:
            if name in inputs and inputs[name] is not None:
                if isinstance(inputs[name], torch.Tensor) and inputs[name].numel() > 0:
                    available_modalities.append(name)
        
        print(f"Available modalities: {available_modalities}")
        
        # Prepare tensors only for available modalities
        prepared_inputs = {}
        grad_vars = []
        grad_names = []
        
        for name in available_modalities:
            tensor = inputs[name].clone().detach().to(self.device)
            # Handle missing WSI case (might have different dimensions)
            if name == 'wsi' and tensor.dim() == 2:
                # If WSI is missing, it might be a zero tensor with different shape
                # Reshape to expected dimensions if needed
                if tensor.shape != (64, 1536):
                    print(f"Warning: WSI has unexpected shape {tensor.shape}")
                    # Pad or reshape as needed based on your model's requirements
                    pass
            
            tensor.requires_grad_(True)
            prepared_inputs[name] = tensor
            grad_vars.append(tensor)
            grad_names.append(name)
        
        # Forward pass with available modalities
        # You need to adapt this based on how your model handles missing modalities
        risk = self._forward_with_missing(prepared_inputs, available_modalities)
        
        # Compute gradients only for available modalities
        gradients = torch.autograd.grad(
            risk, 
            grad_vars,
            grad_outputs=torch.ones_like(risk),
            create_graph=False,
            allow_unused=True
        )
        
        # Compute importance
        importance = {}
        for name, grad in zip(grad_names, gradients):
            if grad is not None:
                if name == 'wsi':
                    grad_norm = grad.view(grad.size(0), -1).norm(dim=1).mean().item()
                else:
                    grad_norm = grad.norm(dim=1).mean().item()
            else:
                grad_norm = 0.0
                print(f"Warning: No gradient for {name}")
            
            importance[name] = grad_norm
        
        # Add zero importance for missing modalities
        all_modalities = ['wsi', 'omic1', 'omic2', 'omic3', 'omic4']
        for name in all_modalities:
            if name not in importance:
                importance[name] = 0.0
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            relative = {k: v/total for k, v in importance.items()}
        else:
            relative = {k: 0.0 for k in importance.keys()}
        
        return {
            'raw': importance,
            'relative': relative,
            'available': available_modalities
        }
    
    def _forward_with_missing(self, inputs, available_modalities):
        """
        Handle forward pass when modalities are missing
        This needs to be customized based on your model
        """
        # Option 1: If your model handles missing data internally
        # Just pass all inputs (with zeros for missing ones)
        
        # Create full input tensors with zeros for missing modalities
        batch_size = next(iter(inputs.values())).shape[0]
        
        omic1 = inputs.get('omic1', torch.zeros(batch_size, self._get_omic_size(0), device=self.device))
        omic2 = inputs.get('omic2', torch.zeros(batch_size, self._get_omic_size(1), device=self.device))
        omic3 = inputs.get('omic3', torch.zeros(batch_size, self._get_omic_size(2), device=self.device))
        omic4 = inputs.get('omic4', torch.zeros(batch_size, self._get_omic_size(3), device=self.device))
        
        # Handle WSI - might need special treatment
        if 'wsi' in inputs:
            wsi = inputs['wsi']
        else:
            # Create dummy WSI (adjust dimensions based on your model)
            wsi = torch.zeros(batch_size, 64, 1536, device=self.device)
        
        # Forward pass
        return self.model.captum(omic1, omic2, omic3, omic4, wsi)
    
    def _get_omic_size(self, idx):
        """Helper to get omic dimension - you'll need to set this"""
        # This should be set from your dataset
        if not hasattr(self, 'omic_sizes'):
            raise AttributeError("Please set self.omic_sizes first")
        return self.omic_sizes[idx]


ig_all = [] 

for i in range(num_splits):
    test_client = split_test[i]
    model = algo_models[current_algo][i]

    model.eval()

    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create the Integrated Gradients wrapper
    ig_split = MultimodalIntegratedGradients(model, omic_sizes=test_client['omic_sizes'])
    ig_all.append(ig_split)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

def extract_importance_data(split_mean_importance_list, modalities):
    """
    Extract importance data from your specific structure
    """    
    n_splits = len(split_mean_importance_list)
    n_modalities = len(modalities)
    
    # Initialize arrays
    raw_importance = np.zeros((n_modalities, n_splits))
    relative_importance = np.zeros((n_modalities, n_splits))
    
    # Extract data
    for split_idx, split_data in enumerate(split_mean_importance_list):
        for mod_idx, modality in enumerate(modalities):
            raw_importance[mod_idx, split_idx] = split_data['raw'][modality]
            relative_importance[mod_idx, split_idx] = split_data['relative'][modality]
    
    return modalities, raw_importance, relative_importance

    
modalities = ['WSI', 'CNV', 'DNAm', 'miRNA', 'mRNA']
n_modalities = len(modalities)

modalities, raw_importance, relative_importance = extract_importance_data(all_mean_importance, modalities)
fig = plot_importances(modalities, raw_importance, relative_importance, n_modalities, n_splits = num_splits)
# Alternative: colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

'''