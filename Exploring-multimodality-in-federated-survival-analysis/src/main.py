#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
import wandb
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_test, get_loaders
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment, save_metrics, print_results
from utils.process_args import _process_args
from utils.tune_parameters import run_tuning

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args):

    # creates scaler and datasets (SUrvivalDataset) for clients, test and val sets
    args.dataset_factory.get_scaler_datasets(args)
    
    # get the loaders of the clients, val sets and test sets included  
    get_loaders(args)
    
    # train and test the model
    if args.tune:
        print('Tuning hyperparameters with Ray Tune')
        run_tuning(args)
    else:
        find_params = False
        new = True
        debug = False
        log_in_same_run = False
        run_name = 'ovy9x3zu'
        run = None

        exclude_keys = ['study', 'task', 'results_dir', 'test_dir', 'val_dir', 'train_dir', 'tune', 'loader_sampler', 'use_nystrom', 'modality', 'fusion']
        config = {k: v for k, v in vars(args).items() if k not in exclude_keys}

        if args.fed_method == 'fedprox':
            wandb.init(project=str(args.study) + "_fed_horizontal_" + args.fed_method + '_zoo-data', name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg) + '_mu: ' + str(args.mu) + '__pat-' +str(args.patience), config=vars(args))
        elif args.fed_method == 'fedopt':
            wandb.init(project=str(args.study) + "_fed_horizontal_" + args.fed_method+ '_zoo-data', name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr-client: ' + str(args.lr_client) + '_decay: ' + str(args.reg) + '_lr-server: ' + str(args.lr_server), config=vars(args))
        elif args.fed_method == 'fedavg':  # TODO

            if log_in_same_run:
                run = wandb.init(project=str(args.study) + "_fed_horizontal_" + args.fed_method + '_debug', name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg) + '_dim-proj-' + str(args.wsi_projection_dim) + '_lay1-' + str(args.encoding_layer_1_dim), id = run_name)
            
            elif debug:
                wandb.init(project=str(args.study) + "_fed_horizontal_" + args.fed_method + '_debug', name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg) + '_dim-proj-' + str(args.wsi_projection_dim) + '_lay1-' + str(args.encoding_layer_1_dim))
            elif new:
                wandb.init(project= str(args.study) + "_fed_horizontal_" + args.fed_method + '_zoo-data', name='split' + str(args.split_num) + '_pat-' + str(args.patience), group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg) + '__pat-' +str(args.patience), config=config)
            elif find_params:
                wandb.init(project="find_new_model-change_dimensions", name='split' + str(args.split_num) + '__v2__', group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg)  + '_dim-proj-' + str(args.wsi_projection_dim) + '_lay1-' + str(args.encoding_layer_1_dim)+ '__v2__', config=vars(args))
            elif args.wsi_projection_dim== 256 and args.encoding_layer_1_dim == 256:
                wandb.init(project=str(args.study) + "_fed_horizontal_" + args.fed_method, name='split' + str(args.split_num) + '__v2__', group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg)  + '_dim-proj-' + str(args.wsi_projection_dim) + '_lay1-' + str(args.encoding_layer_1_dim), config=vars(args))
            else:
                wandb.init(project=str(args.study) + "_fed_horizontal_" + args.fed_method, name='split' + str(args.split_num) + '__v2__', group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg) + '_dim-proj-' + str(args.wsi_projection_dim) + '_lay1-' + str(args.encoding_layer_1_dim) + '__v2__', config=vars(args))
        elif args.fed_method == 'scaffold':
            wandb.init(project=str(args.study) + "_fed_horizontal_" + args.fed_method + '_zoo-data', name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(args.lr) + '_decay: ' + str(args.reg) + '__ep-' + str(args.max_epochs) + '_r0', config=vars(args))

        # training and testing
        run = None if not log_in_same_run else run
        val_cindex, results, test_cindex, test_IBS_list, total_loss, c_lower, c_upper, c_bootstrap_cindexes = _train_test(args, run)

        save_metrics(args, test_cindex, test_IBS_list, total_loss, c_lower, c_upper, c_bootstrap_cindexes)
        print_results(args, results, test_cindex, test_IBS_list, total_loss, c_lower, c_upper, c_bootstrap_cindexes)


if __name__ == "__main__":
    start = timer()

    #----> read the args
    args = _process_args()
    print('Val set global now, new split division, using original dataset, accepted both wsi and omics missing')
    
    #----> Prep
    args = _prepare_for_experiment(args)
    '''
    if args.fed_test_options == 'centralized' and args.fed_method == 'fedavg' and args.wsi_projection_dim == 64 and args.encoding_layer_1_dim == 64:
        if args.lr == 0.00005 and args.reg == 0.0001 and args.split_num == 0:
            print('already done')
            exit(0)
        if args.lr == 0.00005 and args.reg == 0.00001 and args.split_num == 0:
            print('already done')
            exit(0)
            ''''''
    if args.fed_test_options == 'islands' and args.fed_method == 'fedavg' and args.wsi_projection_dim == 64 and args.encoding_layer_1_dim == 64:
        if args.lr == 0.00005 and args.reg == 0.1 and args.split_num == 0:
            print('already done')
            exit(0)
        
        if args.lr == 0.0001 and args.reg == 0.1 and args.split_num == 0:
            print('already done')
            exit(0)'''
    if args.fed_test_options == 'centralized' and args.fed_method == 'fedavg' and args.study == 'tcga_UCEC':
        if args.split_num == 0 and args.reg == 0.0001:
            if args.lr in [0.00005, 0.0001, 0.0005]:
                print('already done')
                exit(0)
        if args.split_num == 0 and args.reg == 0.01:
            if args.lr in [0.00005]:
                print('already done')
                exit(0)
            
    
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed, 
        print_info=False, 
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True if args.modality == "survpath" else False,
        type_of_pathway=args.type_of_path,
        num_clients=args.num_clients,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        train_dir = args.train_dir,
        test_dir=args.test_dir,
        val_dir=args.val_dir,
        fed_option = args.fed_test_options,
        )

    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds\n\n\n' % (end - start))
    wandb.finish()
