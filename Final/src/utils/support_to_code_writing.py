def _summary(test_client, model, modality, loss_fn, clients, survival_train=None):
    r"""
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
    """
    
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






def _calculate_metrics(test_client, clients, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics (c-index, IBS)
    
    Args:
        - test_client : ClientFactory
        - clients : List of ClientFactory
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - IBS_list : List of Float
    """
    
    data = test_client.loader.dataset.metadata["survival_months_dss"]
    bins_original = test_client.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index, lower, upper, bootstrap_cindexes = calculate_cindex((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)
    #c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    IBS_single_list = []
    IBS_all_list = []

    '''
    survival_train  = _extract_survival_metadata(clients)
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.
    IBS_list.append(IBS)
    '''
    
    survival_train  = _extract_survival_metadata(clients = clients)
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.
    IBS_all_list.append(IBS)
    
    for client in clients:
        survival_train  = _extract_survival_metadata(client = client)
        try:
            IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
        except:
            print('An error occured while computing IBS')
            IBS = 0.
        IBS_single_list.append(IBS)    
    
    '''
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
    '''
    return c_index, IBS_single_list, IBS_all_list, lower, upper, bootstrap_cindexes


def calculate_total_loss(model, modality, val_loader, loss_fn):
    r"""
    Run a validation loop on the trained model. Return results of run, total loss, and metrics
    
    Args: 
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
    
    Returns:
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

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
