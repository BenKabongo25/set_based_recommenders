DATASET_PATH=/data/common/RecommendationDatasets/Toys_Amazon14/statements

python main.py \
    --train_data_path ${DATASET_PATH}/train_data.csv \
    --test_data_path ${DATASET_PATH}/test_data.csv \
    --mu 0.0 \
    --neutral_rating 0.0 \
    --use_pairwise_tags \
    --use_item_priors
