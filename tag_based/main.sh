#!/bin/bash

#SBATCH --job-name=tag_based
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

DATASET_NAME=$1
SEED=$2
DATASET_PATH=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/statements
OUTPUT_DIR=/data/common/RecommendationDatasets/exps/TagBased/${DATASET_NAME}/${SEED}

PYTHONPATH=. python baselines/tag_based/main.py \
    --train_data_path ${DATASET_PATH}/train_data.csv \
    --test_data_path ${DATASET_PATH}/test_data.csv \
    --output_dir ${OUTPUT_DIR} \
    --mu 0.0 \
    --neutral_rating 0.0 \
    --use_pairwise_tags \
    --use_item_priors \
    --n_negative_per_positive 99 \
    --seed ${SEED} \
    --use_clustered_tags \
    --all_tags_path ${DATASET_PATH}/aspects.json \
    --selected_tags_index_path ${DATASET_PATH}/selected_aspects_index.json \
    --tag2cluster_index ${DATASET_PATH}/aspect2cluster_id.json \
