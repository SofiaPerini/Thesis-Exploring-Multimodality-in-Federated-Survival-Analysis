from __future__ import print_function, division
from cProfile import label
import os
os.environ["DGL_LOAD_GRAPHBOLT"] = "0"
import pdb
from unittest import case
import pandas as pd
import dgl 
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils.general_utils import _series_intersection
#from clean_datasets import clean_datasets_label, clean_datasets_clinical, clean_datasets_omics

from src.zoorvival.data import load_tcga_data

ALL_MODALITIES = ['rna_clean.csv'] 
BASE = "src/" 


class ClientFactory:
    def __init__(self,
        client_dir,
        client_id, 
        ids_dir = None,
        dataset_path = None,
        ):
        r'''
        Class to store information only relevant to a client: store the datasets of the client, other info specific for the client, and so on...

        Args:
            - client_dir : String      # path to the client
            - client_id : Int 
            - ids_dir : List         # directory to the list of ids that make up the val/train test for normal clients or test set for the test client

        Returns:
            - None
        '''

        self.client_dir = client_dir
        self.client_id = client_id
        self.ids = None
        # save file with list of ids for each client
        path = os.path.join(client_dir, ids_dir)
        self.ids = pd.read_csv(path)    

        self.dataset_path = dataset_path   

        # other variables defined later:
        self.all_modalities = None   # omics data 
        self.label_data = None
        self.patients_df = None
        self.bins = None
        self.patient_dict = None
        self.num_classes = None
        self.label_dict = None
        self.patient_data = None
        self.patient_cls_ids = None
        self.slide_cls_ids = None
        self.clinical_data = None
        self.signatures = None
        self.omic_names = None
        self.omic_sizes = None
        #self.test_ids = None
        self.dataset = None
        self.scaler = None            ## optional, test set does not have scaler
        self.loader = None
        self.val_client = None
        #self.val_loader = None
        #self.val_dataset = None



class SurvivalDatasetFactory:

    def __init__(self,
        study,
        label_file, 
        omics_dir,
        seed, 
        print_info, 
        n_bins, 
        label_col, 
        eps = 1e-6,
        num_patches = 4096,
        is_mcat = False,
        is_survpath = True,
        type_of_pathway = "combine",
        num_clients = 1,
        dataset_path = None,
        split_path = None,
        train_dir = None,
        test_dir = None,
        val_dir = None,
        fed_option = 'federated',
        ):
        r"""
        Initialize the factory to store information relative to the experiment, such as the general directories to the dataset files, 
        arguments relative to the training, the clients and the test (as clients).

        most of them come from args 

        Args:
            - study : String                     #taken from args
            - label_file : String                Path to csv with labels, only general path info, valid for all clients
            - omics_dir : String                 Path to dir with omics csv for all modalities
            - seed : Int
            - print_info : Boolean
            - n_bins : Int
            - label_col: String
            - eps : Float                           # 1e-6
            - num_patches : Int 
            - is_mcat : Boolean
            - is_survapth : Boolean 
            - type_of_pathway : String
            - num_clients : Int
            - dataset_path : String            Path to the clients folders, only general path info, valid for all clients
            - rounds : Int                     used by fed avg
            - train_dir : String               Path to the train set folder
            - test_dir : String                Path to the test set folder
            - val_dir                          Path to the val set folder

        Returns:
            - None
        """

        #---> self
        self.study = study
        self.label_file = label_file
        self.omics_dir = omics_dir
        self.seed = seed
        self.print_info = print_info
        #self.train_ids, self.val_ids  = (None, None)
        self.data_wsi = None
        self.label_col = label_col
        self.n_bins = n_bins
        self.num_patches = num_patches
        self.is_mcat = is_mcat
        self.is_survpath = is_survpath
        self.type_of_path = type_of_pathway

        # clients info
        self.num_clients = num_clients
        self.clients = []
        self.dataset_path = dataset_path
        self.split_path = split_path
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.test_client = None
        self.val_client = None

        if self.label_col == "survival_months":
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship"
        elif self.label_col == "survival_months_pfi":
            self.survival_endpoint = "PFI"
            self.censorship_var = "censorship_pfi"
        elif self.label_col == "survival_months_dss":
            self.survival_endpoint = "DSS"
            self.censorship_var = "censorship_dss"
        elif self.label_col == "survival_months_os":    
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship_os"
        
        # all_clients list includes all the clients and the test and val clients, used to simplify computation 
        all_clients = []

        # create the ClientFactory objects for each client
        for i in range(self.num_clients):
            client = ClientFactory(
                client_dir = os.path.join(self.split_path + "/client_{}".format(i)) if fed_option != 'centralized' else os.path.join(self.split_path + "/client_{}_cent".format(i)),
                client_id = i,
                ids_dir = self.train_dir,
                dataset_path = self.dataset_path,
            )
            self.clients.append(client)
            all_clients.append(client)
        
        # add the test set as a client to SurvivalDatasetFactory
        client_test = ClientFactory(
            client_dir = os.path.join(self.split_path),
            client_id = 'test', 
            ids_dir = self.test_dir,
            dataset_path = self.dataset_path,
            )
        self.test_client = client_test
        all_clients.append(client_test)

        # add the val set as a client to SurvivalDatasetFactory
        client_val = ClientFactory(
            client_dir = os.path.join(self.split_path),
            client_id = 'val', 
            ids_dir = self.val_dir,
            dataset_path = self.dataset_path,
            )
        self.val_client = client_val
        all_clients.append(client_val)


        # read the whole datasets and save it somewhere to avoid reading for every client
        self.read_full_datasets()

        # testing
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Auto-detect terminal width
        pd.set_option('display.max_colwidth', None)  # Show full column content
        
        for client in all_clients: 
            #---> process omics data
            self._setup_omics_data(client)    # start with biological molecules info (get self.self.all_modalities[modality.split('_')[0]]    from datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY})
            
            #---> labels, metadata, patient_df
            self._setup_metadata_and_labels(eps, client)    # get client.label_data from  (datasets_csv)clients/client_i/metadata/tcga_${STUDY}.csv -- calls function to clean the case_ids not present in the wsi database

            #---> prepare for weighted sampling
            self._cls_ids_prep(client)

            #---> load all clinical data 
            self._load_clinical_data(client)        # get self.clinical_data from datasets_csv/clinical_data/${STUDY}_clinical.csv
            
            #---> summarize, print useful info if print_info == True
            self._summarize(client)

            #---> read the signature files for the correct model/ experiment
            if self.is_mcat:
                self._setup_mcat(client)   # 6 functional groups required to run MCAT baseline
            elif self.is_survpath:
                self._setup_survpath(client)  # uploads signatures (pathways)
            else:
                client.omic_names = []
                client.omic_sizes = []
           
        # all omic_sizes and omic_names are the same, depend on data not realated to patients, will be saved as args later for the model  
        print('Dataset_factory and clients created')


    def read_full_datasets(self):
        r'''
        Read the full datasets and save them in dataset_factory parameters.
        Used to only read once the full datasets and avoid repeating the action for every client

        Args:
            - self
        
        Returns:
            - None
        '''

        # omics data
        self.all_modalities = {}
        for modality in ALL_MODALITIES:       ## ALL_MODALITIES = ['rna_clean.csv'] in xena
            self.all_modalities[modality.split('_')[0]] = pd.read_csv(
                os.path.join(self.dataset_path, self.omics_dir, modality),    # omics_dir attribute of args
                engine='python', 
                index_col=0
            )
        
        # label data
        label_data_dir = os.path.join(self.dataset_path, self.label_file)
        self.label_data = pd.read_csv(label_data_dir, low_memory=False)

        # clinical data
        path_to_data = self.dataset_path + "/clinical_data/{}_clinical.csv".format(self.study)
        self.clinical_data = pd.read_csv(path_to_data, index_col=0)

        # pathway compositions and other extra data -- signatures - only read once
        # for the datasets already present in the original code, the signatures are shared, depend only on type of path. For other two datasets are specific
        where = self.type_of_path
        if self.type_of_path == 'other':
            where = self.study
        print('where to get the signatures: ', self.dataset_path + "/metadata/{}_signatures.csv".format(where))
        self.signatures = pd.read_csv(self.dataset_path + "/metadata/{}_signatures.csv".format(where))
        print('shape of signatures: ', self.signatures.shape)



    def _setup_mcat(self, client=None):
        r"""
        Process the signatures for the 6 functional groups required to run MCAT baseline
        
        Args:
            - self 
            -  client : ClientFactory
        
        Returns:
            - None 
        
        """
        client.signatures = pd.read_csv(client.client_dir + "/metadata/signatures.csv")
        client.omic_names = []
        for col in client.signatures.columns:
            omic = client.signatures[col].dropna().unique()   # keep only unique and not na
            omic = sorted(_series_intersection(omic, client.all_modalities["rna"].columns))  # function from utils; Returns insersection of two sets
            client.omic_names.append(omic)
        client.omic_sizes = [len(omic) for omic in client.omic_names]


    def _setup_survpath(self, client=None):

        r"""
        Process the signatures for the 331 pathways required to run SurvPath baseline. Also provides functinoality to run SurvPath with 
        MCAT functional families (use the commented out line of code to load signatures)
        client.signatures contains the signature file, the dataset
        client.omic_names contains the list of the attributes (name of the columns) of the signatures
        client.omic_sizes contains the list of the sizes (rows) for each column
        
        Args:
            - self 
            - client : ClientFactory
        
        Returns:
            - None 
        
        """

        # for running survpath with mcat signatures 
        # self.signatures = pd.read_csv("./datasets_csv/metadata/signatures.csv")
        
        # running with hallmarks, reactome, and combined signatures
        #client.signatures = pd.read_csv(self.dataset_path + "/metadata/{}_signatures.csv".format(self.type_of_path))
        client.signatures = self.signatures
        
        client.omic_names = []
        for col in client.signatures.columns:
            if col == 'Unnamed: 0':
                continue
            omic = client.signatures[col].dropna().unique()
            if len(omic) > 0:
                omic = sorted(_series_intersection(omic, client.all_modalities["rna"].columns))  # function from 
                client.omic_names.append(omic)
        client.omic_sizes = [len(omic) for omic in client.omic_names]
        #print(f'Client {client.client_id}, omic sizes tot number: {len(client.omic_sizes)}')
        print(f'Client {client.client_id}, omic sizes: {client.omic_sizes}')
        #print(f'Client {client.client_id}, omic names: {client.omic_names}')

            

    def _load_clinical_data(self, client=None):
        r"""
        Load the clinical data for the patient. It has grade, stage, etc. Saves it in client.clinical_data
        
        Args:
            - self 
            - client : ClientFactory
        
        Returns:
            - None
            
        """
        # self.study from the args
        #path_to_data = client.client_dir + "/clinical_data/{}_clinical.csv".format(self.study)
        #client.clinical_data = pd.read_csv(path_to_data, index_col=0)
        
        split = []
        if client.client_id == 'test' or client.client_id == 'val':
            split = client.ids[client.client_id]
        else:
            split = client.ids['train']
        client.clinical_data = self.clinical_data[self.clinical_data['case_id'].isin(split)].reset_index(drop=True)
        #print(f'Client {client.client_id}, clinical data: \n {client.clinical_data[:3]}')


    def _setup_omics_data(self, client=None):
        r"""
        Read the csv with the omics data (biological molecules) in os.path.join(client.client_dir, self.omics_dir, modality) 
        and saves it in the client   client.all_modalities
        
        Args:
            - self
            - client : ClientFactory
        
        Returns:
            - None
        
        """
        '''
        client.all_modalities = {}
        for modality in ALL_MODALITIES:       ## ALL_MODALITIES = ['rna_clean.csv'] in xena
            client.all_modalities[modality.split('_')[0]] = pd.read_csv(
                os.path.join(client.client_dir, self.omics_dir, modality),    # omics_dir attribute of args
                engine='python', 
                index_col=0
            )'''

        client.all_modalities = {}
        for modality in ALL_MODALITIES:       ## ALL_MODALITIES = ['rna_clean.csv'] in xena
            mod = modality.split('_')[0]
            split = []
            if client.client_id == 'test' or client.client_id == 'val':
                split = client.ids[client.client_id]
            else:
                split = client.ids['train']
            client.all_modalities[mod] = self.all_modalities[mod][self.all_modalities[mod].index.isin(split)]
            #print(f'Client {client.client_id}, omics data {mod}: \n {client.all_modalities[mod].head(3)}')
            


    def _setup_metadata_and_labels(self, eps, client=None):
        r"""
        Process the metadata required to run the experiment (label_data from lable_file) and saves in client.label_data.
        Minor clean up of the labels and discretization of the data to obtain a classification problem.
        Set up patient dicts to store slide ids per patient.
        Get label dict.
        
        Args:
            - self
            - eps : Float 
            - client : ClientFactory
        
        Returns:
            - None 
        
        """

        #---> read labels from label_file dir
        
        #label_data_dir = os.path.join(client.client_dir, self.label_file)
        #client.label_data = pd.read_csv(label_data_dir, low_memory=False)
        split = []
        if client.client_id == 'test' or client.client_id == 'val':
            split = client.ids[client.client_id]
        else:
            split = client.ids['train']
        client.label_data = self.label_data[self.label_data['case_id'].isin(split)].reset_index(drop=True)
        client.label_data = client.label_data.drop(columns=['Unnamed: 0'])
        #print(f'Client {client.client_id}, label data: \n {client.label_data.head(3)}')
        
        #---> minor clean-up of the labels 
        uncensored_df, uncensored_df_whole = self._clean_label_data(client)  # obtain the uncensored patients (c=0, death observed)
        
        #---> create discrete labels
        self._discretize_survival_months(eps, uncensored_df, uncensored_df_whole, client) # obtain classification problem
    
        #---> get patient info, labels, and metada
        self._get_patient_dict(client)  # for each aptient get key of slide ids, the wsi associated to the patient
        self._get_label_dict(client)
        self._get_patient_data(client)
        

    def _clean_label_data(self, client=None):
        r"""
        Clean the metadata. For breast, only consider the IDC subtype.
        Ignore all labels were subtype of oncotree_code is not IDC, if IDC is present; drop duplicates in labels;
        keep only those were the censorship_var is less than 1 --> (c=0, observed death), uncernsored patients
        
        Args:
            - self 
            - client : ClientFactory
        
        Returns:
            - None  Uncensored_df
            
        """

        if "IDC" in client.label_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            client.label_data = client.label_data[client.label_data['oncotree_code'] == 'IDC']
            self.label_data = self.label_data[self.label_data['oncotree_code'] == 'IDC']

        client.patients_df = client.label_data.drop_duplicates(['case_id']).copy()
        uncensored_df = client.patients_df[client.patients_df[self.censorship_var] < 1]

        self.patients_df = self.label_data.drop_duplicates(['case_id']).copy()
        uncensored_df_whole = self.patients_df[self.patients_df[self.censorship_var] < 1]

        return uncensored_df, uncensored_df_whole


    def _discretize_survival_months(self, eps, uncensored_df, uncensored_df_whole, client=None):
        r"""
        This is where we convert the regression survival problem into a classification problem. We bin all survival times into 
        quartiles and assign labels to patient based on these bins.
        Add to client.patients_df a column with the newly found label
        clients.bin will contain the bins used to discretize
        
        Args:
            - self
            - eps : Float 
            - uncensored_df : pd.DataFrame
            - client : ClientFactory
        
        Returns:
            - None 
        
        """
        # cut the data into self.n_bins (4= quantiles)   # n_bins = args.n_classes
        # consider cut of data according to whole data
        disc_labels, q_bins = pd.qcut(uncensored_df_whole[self.label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = self.label_data[self.label_col].max() + eps
        q_bins[0] = self.label_data[self.label_col].min() - eps
        #print(f'Client {client.client_id}, quantile bins: {q_bins}')

        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        # now assign patients of specific client to general bins
        disc_labels, q_bins = pd.cut(client.patients_df[self.label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        # returns An array-like object representing the respective bin for each value of x; and The computed or specified bins. For scalar or sequence bins, this is an ndarray with the computed bins
        client.patients_df.insert(2, 'label', disc_labels.values.astype(int))
        client.bins = q_bins
        # patient_df does not have duplicates (case_ids)
        #print(f'Client {client.client_id}, discretized labels: \n {client.patients_df[["case_id", self.label_col, "label"]][:5]}')
        

    def _get_patient_data(self, client=None):
        r"""
        Final patient_data is just the clinical metadata + label for the patient ('case_id') 
        client.patient_data contains a dictionary with key: case_id of patient, and label: the label of the patient (class created during discretization)
        
        Args:
            - self 
            - client : ClientFactory
        
        Returns: 
            - None
        
        """
        patients_df = client.label_data[~client.label_data.index.duplicated(keep='first')] 
        patient_data = {'case_id': patients_df["case_id"].values, 'label': patients_df['label'].values} # only setting the final data to self
        client.patient_data = patient_data


    def _get_label_dict(self, client=None):
        r"""
        For the discretized survival times and censorship, we define labels and store their counts.
        Associate a class for each bin-censorship option, and put the new class as 'label' of label_data
        Add to label_data in 'label' the key_count (mapping of bin-censorship possible pair to a int) corresponding to the bin-censorship pair of the patient
        client.label_dict will contain all the possible mappings of bin-censorship possible pair to a int (class)
        client.num_classes will contain the lenght of the label_dict, the number of classes created

        Args:
            - self 
            - client : ClientFactory
        
        Returns:
            - self 
        
        """

        label_dict = {}
        key_count = 0
        for i in range(len(client.bins)-1):  # obtain label_dict = {(0-5,0) : 0; (0-5,1): 1; (5-10,0): 2; ...}? to len(self_bins -2)
            for c in [0, 1]:
                label_dict.update({(i, c):key_count})
                key_count+=1

        for i in client.label_data.index: # for each sample, i is the index
            key = client.label_data.loc[i, 'label'] 
            client.label_data.at[i, 'disc_label'] = key    # Access a single value for a row/column label pair.
            censorship = client.label_data.loc[i, self.censorship_var]
            key = (key, int(censorship))
            client.label_data.at[i, 'label'] = label_dict[key]  # put the corresponding key_count in 'label' of label_data; give class for each bin-censorship option

        client.num_classes = len(label_dict)
        client.label_dict = label_dict


    def _get_patient_dict(self, client=None):
        r"""
        For every patient store the respective slide ids (key for wsi(s)) in self.patient_df where patient_df = {case_id of patient: [list of slide_id of patient]; ...}

        at the beginning:
        client.patients_df = client.label_data.drop_duplicates(['case_id']).copy()  plus add new column for the discretization
        client.label_data = pd.read_csv(self.label_file, low_memory=False); label_file from args

        Args:
            - self 
            - client : ClientFactory
        
        Returns:
            - None

        at the end:   client.label_data = client.patients_df
        """
    
        patient_dict = {}
        temp_label_data = client.label_data.set_index('case_id')  # create an index for the df using the column 'case_id'
        for patient in client.patients_df['case_id']:
            slide_ids = temp_label_data.loc[patient, 'slide_id']  # Access a group of rows and columns by label(s) or a boolean array
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})
        client.patient_dict = patient_dict
        client.label_data = client.patients_df
        client.label_data.reset_index(drop=True, inplace=True)
        # label data now has no duplicates of case_id, but the mutiple slide_ids are saved in patient_dict


    def _cls_ids_prep(self, client):  
        r"""
        Find which patient/slide belongs to which label and store the label-wise indices of patients/ slides

        Creates self.patient_cls_ids: list of sample patients that have as label the same class as the index except for 0; 
        each row = class contains the index of the patients with that label (class)
        Creates self.slide_cls_ids: list of sample slides that have as label the same class as the index except for 0
        one patient/slide for each class -- same but with index of slide_ids

        Args:()
            - self 
            - client : ClientFactory
        
        Returns:
            - None

        """
        client.patient_cls_ids = [[] for i in range(client.num_classes)]   #create a 'matrix', one element array for each range(self.num_classes)
        # Find the index of patients for different labels
        for i in range(client.num_classes):  # for each classification class
            client.patient_cls_ids[i] = np.where(client.patient_data['label'] == i)[0]   # equal to np.asarray(condition).nonzero(): Convert the input to an array (asarray); Return the indices of the elements that are non-zero
            # takes index of samples that has class i as 'label' and that doesn't have 0 as a value

        # Find the index of slides for different labels
        client.slide_cls_ids = [[] for i in range(client.num_classes)]
        for i in range(client.num_classes):
            client.slide_cls_ids[i] = np.where(client.label_data['label'] == i)[0]
        
        

    def _summarize(self, client=None):
        r"""
        If self.print_info == True:
        Summarize which type of survival you are using, number of cases and classes
        print useful information about columns; num of cases (samples), classes 
        Args:
            - self 
            - client : ClientFactory
        
        Returns:
            - None 
        
        """

        if self.print_info:
            print('client number: {}'.format(client.client_id))
            print("label column: {}".format(self.label_col))
            print("number of cases for client {}".format(len(client.label_data)))
            print("number of classes for client: {}".format(client.num_classes))

    '''
    def _patient_data_prep(self):
        patients = np.unique(np.array(self.label_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.label_data[self.label_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.label_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}
    '''


    def get_scaler_datasets(self, args):
        r'''
        Generates a scaler from the whole dataset and uses it to normalize the datasets of the clients, test and val set.
        return for each client /test and val included) a dataset, saved into client.dataset.

        Args:
            - self
            - args : argspace.Namespace
        
        Return:
            - None

        '''

        # create a scaler for the clients and test val sets considering the whole dataset (omics data) not federated
        scaler = self.get_scaler_from_df()

        # for the val client, get val dataset and scaler
        val_dataset = self.get_split_from_df(args, client=args.dataset_factory.val_client, scaler=scaler, is_test=True, is_val=True)
        self.val_client.dataset = val_dataset

        # for each client get train dataset and scaler, and the validation set dataset
        for client in self.clients:
            dataset, scaler = self.get_split_from_df(args, client=client, scaler=scaler, is_test=False, is_val=False)
            client.dataset = dataset
            client.scaler = scaler
            #print(f'Client {client.client_id}: \n tot train data: {client.dataset.metadata.shape} \n train data: {client.dataset.metadata[:5]} \ntrain set label data: \n {client.dataset.metadata["case_id"].value_counts()}\n val set label data: \n {client.val_dataset.metadata["case_id"].value_counts()}')
            #print()

        # for the test set
        dataset_t = self.get_split_from_df(args, client=args.dataset_factory.test_client, scaler=scaler, is_test=True)
        self.test_client.dataset = dataset_t
        
        '''
        # for the val set
        dataset_v = self.get_split_from_df(args, client=args.dataset_factory.val_client, scaler=scaler, is_test=True, is_val=True)
        self.val_client.dataset = dataset_v
        '''

        print("Created train, test and val datasets")
    
    
    
    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        _, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins
    

    def _get_scaler(self, data):
        r"""
        Define the scaler for training dataset. Use the same scaler for validation set
        MinMaxSCaler: Transform features by scaling each feature to a given range.
        This estimator scales and translates each feature individually such that it is in the given range on the training set
        
        Args:
            - self 
            - data : np.array

        Returns: 
            - scaler : MinMaxScaler
        
        """
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        return scaler
        
    
    def _apply_scaler(self, data, scaler):
        r"""
        Given the datatype and a predefined scaler, apply it to the data 
        
        Args:
            - self
            - data : np.array 
            - scaler : MinMaxScaler 
        
        Returns:
            - data : np.array """
        
        # find out which values are missing
        zero_mask = data == 0

        # transform data
        transformed = scaler.transform(data)
        data = transformed

        # rna -> put back in the zeros 
        data[zero_mask] = 0.
        
        return data


    def get_scaler_from_df(self):
        r"""
        Obtain a scaler from the whole dataset using omics data. We are using the case_ids, 
        so it would be better to have a second version of the dataset that contains different values in their place

        Args:
            - self  

        Returns:
            - scaler : MinMaxScaler
        
        """

        scaler = {}

        for key in self.all_modalities.keys():   # contains modalities for omics data
            
            filtered_df = self.all_modalities[key]  #raw_data_df[mask]   # only mask
            filtered_df = filtered_df[~filtered_df.index.duplicated()] # drop duplicate case_ids
            filtered_df.reset_index(inplace=True, drop=True)
 
            # flatten the df into 1D array (make it a column vector)
            flat_df = filtered_df.values.flatten().reshape(-1, 1)    #df_for_norm.values.flatten().reshape(-1, 1)
            
            # get scaler
            scaler_for_data = self._get_scaler(flat_df)

            # store scaler
            scaler[key] = scaler_for_data

        return scaler


    def get_split_from_df(self, args, client, scaler, is_test=False, is_val=False, valid_cols=None):
        r"""
        Apply scaler on datasets -- Standardize features by removing the mean and scaling to unit variance.
        Create new dataset formatted SurvivalDataset for the data of the client, returns it

        Args:
            - self 
            - args : argspace.Namespace
            - client : ClientFactory        can be a client or the test set (also saved as a client)
            - scaler : MinMaxScaler
            - is_test : Boolean             indicates if it's a test set or a val set
            - is_val : Boolean              indicates if it's a val set or not
            - valid_cols : List 

        Returns:
            - SurvivalDataset 
            - Optional: scaler (MinMaxScaler)
        
        """
        
        split = None
        where = 'train'
        if is_test:
            where = 'test'
            if is_val:
                where = 'val'

        split = client.ids[where]
        split = split.dropna().reset_index(drop=True)
        print("Total number of patients for the client {} in the set {}: {}".format(client.client_id, where, len(split))) 
        
        '''
        if is_test and not is_val:
            # read the test.csv or val.csv and save it in test_ids
            split = client.ids['test']
            print("Total number of patients in the {} set: {}".format('test', split))
        elif is_val:
            split = client.ids['val']
            split = split.dropna().reset_index(drop=True)
            print("Total number of patients in the validation set for the client {}: {}".format(client.client_id, len(split))) 
        else:
            #split = np.union1d(client.label_data['case_id'], client.all_modalities['rna'].index)  # get all case_ids from the client
            #split = np.union1d(split, client.clinical_data['case_id'])
            #split = np.unique(split)
            split = client.ids['train']
            split = split.dropna().reset_index(drop=True)
            print("Total number of patients in the client {}: {}".format(client.client_id, len(split))) 
        '''

        mask = client.label_data['case_id'].isin(split.tolist())
        df_metadata_slide = client.label_data.loc[mask, :].reset_index(drop=True)   # take the corresponding samples from label data
        
        # select the rna, meth (drugs?), mut (mutations in rna), cnv (type of deviation in rna) data for this split
        omics_data_for_split = {}
        for key in client.all_modalities.keys():   # contains modalities for omics data
            
            raw_data_df = client.all_modalities[key]
            mask = raw_data_df.index.isin(split.tolist())   # new mask between split index and modality data
            
            filtered_df = raw_data_df[mask]    # raw_data_df[mask]   # only mask
            filtered_df = filtered_df[~filtered_df.index.duplicated()] # drop duplicate case_ids
            filtered_df["temp_index"] = filtered_df.index
            filtered_df.reset_index(inplace=True, drop=True)

            clinical_data_mask = client.clinical_data.case_id.isin(split.tolist())  # same with clinical data, new mask
            clinical_data_for_split = client.clinical_data[clinical_data_mask]
            clinical_data_for_split = clinical_data_for_split.set_index("case_id")   # new index
            clinical_data_for_split = clinical_data_for_split.replace(np.nan, "N/A")  # keep na data

            # from metadata (label data) and clinical data drop any cases that are not in filtered_df
            # keep patients in label data that are also in filtered df (patients that don't have omics data)
            '''
            mask = [True if item in list(filtered_df["temp_index"]) else False for item in df_metadata_slide.case_id]
            df_metadata_slide = df_metadata_slide[mask]
            df_metadata_slide.reset_index(inplace=True, drop=True)  
            '''

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
                scaler_for_data = scaler[key]
                #scaler_for_data = scaler

                # normalize 
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                # change 1D to 2D
                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                # add in case_ids
                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

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
                scaler_for_data = scaler[key]

                # normalize 
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                # change 1D to 2D
                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                # add in case_ids
                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

                # store scaler
                scaler[key] = scaler_for_data
                
            omics_data_for_split[key] = filtered_normed_df

        if not is_test:
            sample = True
        else:
            sample = False

        # get zoorvival project
        project = args.study[5:].upper()  # from tcga_brca to BRCA
        db = load_tcga_data(project)
            
        split_dataset = SurvivalDataset(
            client_id = client.client_id,
            study_name = args.study,
            modality = args.modality,
            patient_dict = client.patient_dict,
            metadata = df_metadata_slide,    # get label data
            omics_data_dict = omics_data_for_split,  # get normalized omics dat
            data_wsi = db,    # get wsi images
            num_classes = client.num_classes,
            label_col = self.label_col,
            censorship_var = self.censorship_var,
            valid_cols = valid_cols,
            is_test = is_test == False,   ## ????
            clinical_data = clinical_data_for_split,  # get clinicla data
            num_patches = self.num_patches,
            omic_names = client.omic_names,
            sample = sample
            )

        if is_test:
            return split_dataset    # returns the SurvivalDataset
        else:
            return split_dataset, scaler
        
    
    def __len__(self):
        return len(self.label_data) 
    

class SurvivalDataset(Dataset):

    def __init__(self,
        client_id,
        study_name,
        modality,
        patient_dict,
        metadata, 
        omics_data_dict,
        data_wsi, 
        num_classes,
        label_col = "survival_months_DSS",
        censorship_var = "censorship_DSS",
        valid_cols = None,
        is_test = False,
        clinical_data = -1,
        num_patches = 4000,
        omic_names = None,
        sample = True,
        ): 

        super(SurvivalDataset, self).__init__()

        #---> self
        self.client_id = client_id
        self.study_name = study_name
        self.modality = modality
        self.patient_dict = patient_dict
        self.metadata = metadata 
        self.omics_data_dict = omics_data_dict
        self.data_wsi = data_wsi
        self.num_classes = num_classes
        self.label_col = label_col
        self.censorship_var = censorship_var
        self.valid_cols = valid_cols
        self.is_test = is_test
        self.clinical_data = clinical_data
        self.num_patches = num_patches
        self.omic_names = omic_names
        self.num_pathways = len(omic_names)
        self.sample = sample

        # for weighted sampling
        self.slide_cls_id_prep()
    
    def _get_valid_cols(self):
        r"""
        Getter method for the variable self.valid_cols 
        """
        return self.valid_cols

    def slide_cls_id_prep(self):
        r"""
        For each class, find out how many slides do you have 
        
        Args:
            - self 
        
        Returns: 
            - None
        
        """

        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.metadata['label'] == i)[0] # get list of indexes where label==i
            #list of samples that have as label the same class as the index except for value 0 
         
    def __getitem__(self, idx):
        r"""
        Given the modality, return the correctly transformed version of the data
        You don't explicitly call it, it's called automatically by the DataLoader when you iterate over it: 
        when you iterate, the DataLoader needs to fetch samples, for each sample in the batch, it calls dataset.__getitem__(index)
        
        Args:
            - idx : Int 
        
        Returns:
            - variable, based on the modality 
        
        """
        
        label, event_time, c, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)

        if self.modality in ['omics', 'snn', 'mlp_per_path']:

            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            
            return (torch.zeros((1,1)), omics_tensor, label, event_time, c, clinical_data)
        
        # what is the difference between tmil_abmil and transmil_wsi
        elif self.modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:

            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            patch_features, mask = self._load_wsi_embs_from_path(self.data_wsi, slide_ids)
            
            #@HACK: returning case_id, remove later
            return (patch_features, omics_tensor, label, event_time, c, clinical_data, mask)

        elif self.modality in ["coattn", "coattn_motcat"]:
            
            patch_features, mask = self._load_wsi_embs_from_path(self.data_wsi, slide_ids)

            omic1 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[0]].iloc[idx])
            omic2 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[1]].iloc[idx])
            omic3 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[2]].iloc[idx])
            omic4 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[3]].iloc[idx])
            omic5 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[4]].iloc[idx])
            omic6 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[5]].iloc[idx])

            return (patch_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data, mask)
        
        elif self.modality == "survpath":
            patch_features, mask = self._load_wsi_embs_from_path(self.data_wsi, slide_ids)

            omic_list = []
            #print(self.num_pathways)
            for i in range(self.num_pathways):
                # the omics data may not be present for the considered index, return None
                try:
                    omic_list.append(torch.tensor(self.omics_data_dict["rna"][self.omic_names[i]].iloc[idx]))
                except:
                    omic_list = None
                    break
            
            #if patch_features is None and omic_list is None:
                #print(f'{case_id} has no WSI embeddings and no omics data')
            if omic_list is None or len(omic_list)==0:
                print(f'{case_id} has no omics data')
            #elif patch_features is None:
                #print(f'{case_id} has no WSI embeddings')
            #print('omic: ', omic_list)
            #print('label', label)
            #print('patch_features: ', patch_features, 'omic_list: ', omic_list, 'label: ', label.item())
            return (patch_features if patch_features is not None else None, omic_list if omic_list is not None else None, label, event_time, c, clinical_data, mask if mask is not None else None)
        
        else:
            raise NotImplementedError('Model Type [%s] not implemented.' % self.modality)


    def get_data_to_return(self, idx):
        r"""
        Collect all metadata and slide data to return for this case ID 
        
        Args:
            - idx : Int 
        
        Returns: 
            - label : torch.Tensor
            - event_time : torch.Tensor
            - c : torch.Tensor
            - slide_ids : List
            - clinical_data : tuple
            - case_id : String
        """

        case_id = self.metadata['case_id'].iloc[idx]
        #print('getting data for index: {}, case id: {}'.format(idx, case_id))
        label = torch.Tensor([self.metadata['disc_label'].iloc[idx]]) # disc - no attribute 'disc_label' in metadata, added with initial processing
        event_time = torch.Tensor([self.metadata[self.label_col].iloc[idx]])
        c = torch.Tensor([self.metadata[self.censorship_var].iloc[idx]])
        slide_ids = self.patient_dict[case_id]
        clinical_data = self.get_clinical_data(case_id)
        #print(f'Case ID: {case_id}, Label: {label.item()}, Event Time: {event_time.item()}, Censorship: {c.item()}, Slide IDs: {slide_ids}, Clinical Data: {clinical_data}')
        return label, event_time, c, slide_ids, clinical_data, case_id
    

    def _load_wsi_embs_from_path(self, data_wsi, slide_ids):   
        """
        Load all the patch embeddings from a list a slide IDs. first change the slide_id name to the one used in the zoorvival ds.
        then, load the embeddings for each slide_id from the data_wsi (which is a zoorvival dataset) and concatenate them using the db.train/test.df_clinical 
        as a support to find the right index for the slide_id in the wsi_embeddings.
        For some patients (case_ids) there are multiple slide_ids. These slide_ids are not present in the zoorvival dataset, so we only upload one slide_id (the one present)

        Args:
            - self 
            - data_wsi : DataFrame
            - slide_ids : List
        
        Returns:
            - patch_features : torch.Tensor 
            - mask : torch.Tensor

        """
        patch_features = []
        zoo_slides = []

        #print(f'Loading WSI embeddings for slide IDs: {slide_ids}')
        # get the right format for the slide_ids: can find them in zoorvival using a short version of patient_id
        for slide in slide_ids:
            id = slide[5:12]   # remove 'TCGA-' and keep the case_id
            zoo_slides.append(id)
        # remove duplicates as to only remain with one slide_id per patient
        zoo_slides = list(set(zoo_slides))
        #print(f"loadinf slide id: {zoo_slides}")
        if len(zoo_slides) > 1:
            print(f"-------  Patient {zoo_slides} has multiple slide IDs {slide_ids} ---- PROBLEM!!!!")

        #print('******* Testing wsi embedding sizes *******')
        #print(f'Loading WSI embeddings for slide: {zoo_slides}')

        # load all slide_ids corresponding for the patient
        for id in zoo_slides:
            wsi_bag = np.zeros((64, 1536))
            #wsi_path = os.path.join(data_wsi, '{}.pt'.format(slide_id.rstrip('.svs')))
            #wsi_bag = torch.load(wsi_path)

            # chec if id is a index in train or test separation for df_clinical
            if id in data_wsi.train.df_clinical.index:
                wsi_idx = data_wsi.train.df_clinical.index.get_loc(id)   # get the integer index from the string index
                wsi_bag = data_wsi.train.wsi_embeddings[wsi_idx]  # get the wsi bag for this index
            elif id in data_wsi.test.df_clinical.index:
                wsi_idx = data_wsi.test.df_clinical.index.get_loc(id)
                wsi_bag = data_wsi.test.wsi_embeddings[wsi_idx] 
            else:  # TODO: 
                # if the wsi is not present, load all zeroes

                #print("PROBLEM! Slide ID {} not found in either train or test dataset.".format(id))
                # go to next slide_id skipping this iteration
                #continue
    
                # if return None:
                return None, None

            # convert numpy array to torch tensor
            if isinstance(wsi_bag, np.ndarray):
                if wsi_bag.ndim == 1:
                    wsi_bag = wsi_bag.reshape(1, -1)
            
            wsi_bag = torch.from_numpy(wsi_bag)
            patch_features.append(wsi_bag)
            #print(f'Loaded WSI embedding for slide ID {id} with shape: {wsi_bag.shape}')  # [64, 1536]

        patch_features = torch.cat(patch_features, dim=0)
        #print(f'Loaded WSI embeddings with shape: {patch_features.shape}')

        if self.sample:  # True for training, false for testing and val
            max_patches = self.num_patches

            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]
            #print(f'Sampled {n_samples} patches for WSI embeddings.')   # 64
            #print(f'Patch features shape after sampling: {patch_features.shape}')   # [64, 1536]
            # make a mask 
            if n_samples == max_patches:
                # sampled the max num patches, so keep all of them
                mask = torch.zeros([max_patches])
            else:
                # sampled fewer than max, so zero pad and add mask
                original = patch_features.shape[0]
                how_many_to_add = max_patches - original
                zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                patch_features = torch.concat([patch_features, zeros], dim=0)
                mask = torch.concat([torch.zeros([original]), torch.ones([how_many_to_add])])
        
        else:
            mask = torch.ones([1])

        return patch_features, mask


    def get_clinical_data(self, case_id):
        """
        Load all the patch embeddings from a list a slide IDs. 

        Args:
            - data_wsi : String 
            - slide_ids : List
        
        Returns:
            - patch_features : torch.Tensor 
            - mask : torch.Tensor

        """
        try:
            stage = self.clinical_data.loc[case_id, "stage"]
        except:
            stage = "N/A"
        
        try:
            grade = self.clinical_data.loc[case_id, "grade"]
        except:
            grade = "N/A"

        try:
            subtype = self.clinical_data.loc[case_id, "subtype"]
        except:
            subtype = "N/A"
        
        clinical_data = (stage, grade, subtype)
        return clinical_data
    

    def getlabel(self, idx):
        r"""
        Use the metadata for this dataset to return the survival label for the case 
        
        Args:
            - idx : Int 
        
        Returns:
            - label : Int 
        
        """
        print('getting label in different function for index: {}'.format(idx))
        label = self.metadata['label'].iloc[idx]
        return label

    def __len__(self):
        return len(self.metadata) 