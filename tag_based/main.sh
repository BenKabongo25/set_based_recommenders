DATASET_NAME=$1
DATASET_PATH=/data/common/RecommendationDatasets/${DATASET_NAME}_Amazon14/statements
OUTPUT_DIR=/data/common/RecommendationDatasets/exps/TagBased/${DATASET_NAME}/

python main.py \
    --train_data_path ${DATASET_PATH}/train_data.csv \
    --test_data_path ${DATASET_PATH}/test_data.csv \
    --output_dir ${OUTPUT_DIR} \
    --mu 0.0 \
    --neutral_rating 0.0 \
    --use_pairwise_tags \
    --use_item_priors
