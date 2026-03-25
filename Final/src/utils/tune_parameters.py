from ray import tune
from ray.tune.schedulers import ASHAScheduler
import wandb

from utils.core_utils import _train_test
from utils.general_utils import save_metrics, print_results


def run_tuning(args):
    
    def tune_trial(config):
        # Each trial trains a model with given hyperparameters

        debug = True
        text = ''
        if debug:
            text = '_debug'

        if args.fed_method == 'fedprox':
            wandb.init(project="fed_horizontal_" + args.fed_method + text, name=f"trial_{tune.get_trial_id()}", group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(config["lr"]) + '_decay: ' + str(config["weight_decay"]), config=config, reinit=True)
        elif args.fed_method == 'fedopt':
            wandb.init(project="fed_horizontal_" + args.fed_method + text, name=f"trial_{tune.get_trial_id()}", group= args.fed_method + '-' + args.fed_test_options+ '_lr: ' + str(config["lr"]) + '_decay: ' + str(config["weight_decay"]), config=config, reinit=True)
        elif args.fe_method == 'fedavg':
            wandb.init(project="fed_horizontal_" + args.fed_method + text, name=f"trial_{tune.get_trial_id()}", group= args.fed_method + '-' + args.fed_test_options+ '_lr: ' + str(config["lr"]) + '_decay: ' + str(config["weight_decay"]), config=config, reinit=True)

        print(f"Starting trial {tune.get_trial_id()} with config: {config}")
        
        val_loss, _, _, _, _ = _train_test(
            args = args,
            lr = config["lr"],
            weight_decay = config["weight_decay"]
        )

        # Report validation metrics to Ray Tune
        tune.report(
            val_loss = val_loss
            #val_cindex=val_metrics["val_cindex"],
            #val_ibs=val_metrics["val_ibs"]
        )

        wandb.finish()


    search_space = {   # TODO
        "lr": tune.choice([0.0001, 0.0005, 0.001]),
        "weight_decay": tune.choice([0.00001, 0.0001, 0.01])
    }

    scheduler = ASHAScheduler(metric="val_loss", mode="min")
    tuner = tune.Tuner(
        tune.with_parameters(tune_trial),
        param_space = search_space,
        tune_config = tune.TuneConfig(
            metric = "val_loss",
            mode = "min",
            scheduler = scheduler,
            num_samples = 10,
            max_concurrent_trials = 2
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config

    print(f"Best config: {best_config}")

    args.lr = best_config["lr"]
    args.reg = best_config["weight_decay"]
    # new wandb run with final best model
    if args.fed_method == 'fedprox':
        wandb.init(project="fed_horizontal", name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(best_config["lr"]) + '_decay: ' + str(best_config["weight_decay"]) + '_mu: ' + str(args.mu), config=vars(args), reinit=True)
    elif args.fed_method == 'fedopt':
        wandb.init(project="fed_horizontal", name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr-client: ' + str(best_config["lr"]) + '_decay: ' + str(best_config["weight_decay"]) + '_lr-server: ' + str(args.lr_server), config=vars(args), reinit=True)
    elif args.fe_method == 'fedavg':
        wandb.init(project="fed_horizontal", name='split' + str(args.split_num), group= args.fed_method + '-' + args.fed_test_options + '_lr: ' + str(best_config["lr"]) + '_decay: ' + str(best_config["weight_decay"]), config=vars(args), reinit=True)


    # Retrain the best model and evaluate on test set
    val_loss, results, test_cindex, test_IBS_list, total_loss = _train_test(
        args = args,
        lr = best_config["lr"],
        weight_decay = best_config["weight_decay"]
    )

    save_metrics(args, test_cindex, test_IBS_list, total_loss)      
    print_results(args, results, test_cindex, test_IBS_list, total_loss)

    wandb.finish()