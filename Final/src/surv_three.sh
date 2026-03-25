#!/bin/bash

STUDY=("BRCA")
DATA_ROOT_DIR='data/TCGA/BRCA' # where are the TCGA features stored?
BASE_DIR="~/SurvPath--1/src/" # where is the repo cloned?
BASE="src"
TYPE_OF_PATH="other" # what type of pathways?  #keep
MODEL="survpath" # what type of model do you want to train?
LAY1=64      ## about pathway tokenization, DIM1 and DIM2, keep
EMB=64
LRS=(0.001)
DECAYS=(0.01)
SPLITS=(0 1 2 3 4)
MUS=(0.001)

for decay in ${DECAYS[@]};
do
    for lr in ${LRS[@]};
    do 
        for split in ${SPLITS[@]};
        do
            for mu in ${MUS[@]};
            do
                python $BASE/main.py \
                    --study tcga_$STUDY --task survival \
                    --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/${STUDY}_gigapath_compressed_embeddings.h5 --label_file metadata/tcga_$STUDY.csv \
                    --omics_dir raw_rna_data/${TYPE_OF_PATH}/$STUDY --results_dir $BASE/results_$STUDY --model_dir $BASE/saved_models/weights_3.pth \
                    --num_clients 4 --dataset_path $BASE/datasets_csv --split_path $BASE/splits_$STUDY/split_$split --split_num $split \
                    --fed_test_options federated --fed_method fedprox --mu $mu \
                    --batch_size 1 --lr $lr --opt radam --reg $decay --max_rounds 300 --patience 32 --lr_pat 20 --test_dir test.csv --val_dir val.csv --train_dir train.csv \
                    --alpha_surv 0.5 --weighted_sample --max_epochs 1 --encoding_dim 1536 \
                    --label_col survival_months_os --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim ${EMB} \
                    --encoding_layer_1_dim ${LAY1} --encoder_dropout 0.25 --loader_sampler 0 --is_save_model 1
            done
        done
    done
done 