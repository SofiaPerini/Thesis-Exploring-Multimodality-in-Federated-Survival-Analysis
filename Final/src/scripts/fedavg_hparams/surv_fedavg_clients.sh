#!/bin/bash

DATA_ROOT_DIR='data/TCGA/BRCA' # where are the TCGA features stored?
BASE_DIR="~/survpath/SurvPath--1/src/" # where is the repo cloned?
BASE="src"
TYPE_OF_PATH="xena" # what type of pathways?  #keep
MODEL="survpath" # what type of model do you want to train?
DIM1=8      ## about pathway tokenization, DIM1 and DIM2, keep
DIM2=16
STUDIES=("brca")
LRS=(0.00005 0.0001 0.0005 0.001)   #keep
DECAYS=(0.00001 0.0001 0.001 0.01)   #keep

for decay in ${DECAYS[@]};
do
    for lr in ${LRS[@]};
    do 
        for STUDY in ${STUDIES[@]};
        do
            python $BASE/main.py \
                --study tcga_${STUDY} --task survival \
                --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/BRCA_gigapath_compressed_embeddings.h5 --label_file metadata/tcga_${STUDY}.csv \
                --omics_dir raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir $BASE/results_${STUDY} \
                --num_clients 4 --clients_path $BASE/clients_fed \
                --fed_test_options federated \
                --batch_size 1 --lr $lr --opt radam --reg $decay --max_rounds 200 --patience 30 --test_dir $BASE/clients_fed/test_set --val_dir train-val_split.csv \
                --alpha_surv 0.5 --weighted_sample --max_epochs 2 --encoding_dim 1536 \
                --label_col survival_months_dss --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                --encoding_layer_1_dim ${DIM1} --encoding_layer_2_dim ${DIM2} --encoder_dropout 0.25
        done 
    done
done 



# save surv_mio.sh
#!/bin/bash

DATA_ROOT_DIR='data/TCGA/BRCA' # where are the TCGA features stored?
BASE_DIR="~/survpath/SurvPath--1/src/" # where is the repo cloned?
BASE="src"
TYPE_OF_PATH="xena" # what type of pathways?  #keep
MODEL="survpath" # what type of model do you want to train?
DIM1=8      ## about pathway tokenization, DIM1 and DIM2, keep
DIM2=16
STUDIES=("brca")
LRS=(0.00005 0.0001 0.0005 0.001)   #keep
DECAYS=(0.00001 0.0001 0.001 0.01)   #keep

for decay in ${DECAYS[@]};
do
    for lr in ${LRS[@]};
    do 
        for STUDY in ${STUDIES[@]};
        do
            python $BASE/main.py \
                --study tcga_${STUDY} --task survival --split_dir $BASE/splits \
                --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/BRCA_gigapath_compressed_embeddings.h5 --label_file metadata/tcga_${STUDY}.csv \
                --omics_dir raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir $BASE/results_${STUDY} \
                --num_clients 4 --clients_path $BASE/clients_4 \
                --batch_size 1 --lr $lr --opt radam --reg $decay --max_rounds 80 --patience 5 --test_dir $BASE/clients_4/test_set --val_dir train-val_split.csv \
                --alpha_surv 0.5 --weighted_sample --max_epochs 2 --encoding_dim 1536 \
                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                --encoding_layer_1_dim ${DIM1} --encoding_layer_2_dim ${DIM2} --encoder_dropout 0.25
        done 
    done
done 