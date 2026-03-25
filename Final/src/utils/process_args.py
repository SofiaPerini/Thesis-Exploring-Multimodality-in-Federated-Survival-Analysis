import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, help='study name')
    parser.add_argument('--task', type=str, choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="hallmarks", choices=["xena", "hallmarks", "combine", 'other'])
    #parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--label_file', type=str, default=None, help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default=None, help='Path to dir with omics csv for all modalities')
    parser.add_argument('--num_patches', type=int, default=4000, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months_dss", help='type of survival (OS, DSS, PFI)')
    parser.add_argument("--wsi_projection_dim", type=int, default=1)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=64)
    #parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)
    parser.add_argument('--num_clients', type=int, default=1, help='number of clients for FL')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the clients data, only general path info, data for all clients')
    parser.add_argument('--split_path', type=str, default=None, help='Path to the split data info, contains the division of the data between train/val/test (divided by clients) for the split')
    parser.add_argument('--split_num', type=int, default=0, help='which split to use')

    #----> split related 
    #parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    #parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    #parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    #parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' 
    #                +'instead of infering from the task and label_frac argument (default: None)')
    #parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')

    parser.add_argument('--test_dir', type=str, default=None, help='Path to the test set folder')
    parser.add_argument('--val_dir', type=str, default=None, help='Path to the val set folder')
    parser.add_argument('--train_dir', type=str, default=None, help='Path to the train set folder')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to the the folder where the model weights are saved during the training')
        
    #----> training related 
    parser.add_argument('--tune', type=bool, default=False, help='enable hyperparameter tuning with ray tune')
    parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg', 'fedprox', 'fedopt', 'scaffold'])
    parser.add_argument('--mu', type=float, default=0.0, help='variable used only by fedprox')
    parser.add_argument('--fed_test_options', type=str, default='federated', choices=['federated', 'centralized', 'islands'])
    parser.add_argument('--max_rounds', type=int, default=1, help='number of training rounds over the clients (default: 1)')
    parser.add_argument('--patience', type=int, default=5, help='number of training rounds with no improvement before the early stopping mechanisms ends the training')
    parser.add_argument('--lr_pat', type=int, default=8, help='number of training rounds with no improvement before lr scheduler reduces the lr')
    parser.add_argument('--max_epochs', type=int, default=2, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_client', type=float, default=1e-4, help='learning rate for the clients (default: 0.0001, equal to lr)')
    parser.add_argument('--lr_server', type=float, default=0.0, help='learning rate for the server for fedopt (default: 0.0)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv"], default='ce',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.0, help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay / L2 (default: 1e-5)')
    #parser.add_argument('--lr_scheduler', type=str, default='cosine')
    #parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--loader_sampler', type=int, default=0, choices=[0, 1], help='1 or 0 id want or not the sampler in the data loader')
    parser.add_argument('--is_save_model', type=int, default=0, choices=[0, 1], help='1 or 0 if the model is to be saved after training or not')
    
    #---> model related
    parser.add_argument('--fusion', type=str, default=None)
    parser.add_argument('--modality', type=str, default="wsi")
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim')
    parser.add_argument('--use_nystrom', action='store_true', default=False, help='Use Nystrom attentin in SurvPath.')

    args = parser.parse_args()

    if args.fed_method == 'fedprox' and args.mu < 0.0:
        print('Error! You are using fedprox without setting mu')
        exit()
        
    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    args.save_model_dir = 'model_weights_' + str(args.study) + '/' + str(args.fed_test_options) + '_' + str(args.fed_method) + '/split_' + str(args.split_num) + '/weights.pth'

    return args
