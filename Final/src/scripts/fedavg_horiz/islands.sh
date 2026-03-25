#!/bin/bash

DATA_ROOT_DIR='data/zoorvival/TCGA/BRCA' # where are the TCGA features stored?
BASE_DIR="~/SurvPath--1/src/" # where is the repo cloned?
BASE="src"
TYPE_OF_PATH="hallmarks" # what type of pathways?  #keep
MODEL="survpath" # what type of model do you want to train?
LAY1=64      ## about pathway tokenization, dimension of first layer
EMB=64       ## dimension of embedding for wsi and pathways
STUDY=("blca")
LRS=(0.0001)
DECAYS=(0.01)
SPLITS=(0)

for decay in ${DECAYS[@]};
do
    for lr in ${LRS[@]};
    do 
        for split in ${SPLITS[@]};
        do
            python $BASE/main.py \
                --study tcga_$STUDY --task survival \
                --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/BRCA_gigapath_compressed_embeddings.h5 --label_file metadata/tcga_$STUDY.csv \
                --omics_dir raw_rna_data/${TYPE_OF_PATH}/$STUDY --results_dir $BASE/results_$STUDY --model_dir $BASE/saved_models/weights.pth \
                --num_clients 4 --dataset_path $BASE/datasets_csv_original --split_path $BASE/splits_original/split_$split --split_num $split \
                --fed_test_options islands --fed_method fedavg \
                --batch_size 1 --lr $lr --opt radam --reg $decay --max_rounds 1 --patience 32 --lr_pat 20 --test_dir test.csv --val_dir val.csv --train_dir train.csv \
                --alpha_surv 0.5 --weighted_sample --max_epochs 300 --encoding_dim 1536 \
                --label_col survival_months_dss --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim ${EMB} \
                --encoding_layer_1_dim ${LAY1} --encoder_dropout 0.25 --loader_sampler 0
        done
    done
done 