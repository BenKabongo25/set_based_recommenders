# Ben Kabongo
# August 2025


import argparse
import ast
import json
import os
import pandas as pd

from recommender import TagBaseRecommender
from utils import normalize_overall_rating, evaluate_ranking_metrics


def main(config):
    train_df = pd.read_csv(config.train_data_path)
    test_df = pd.read_csv(config.test_data_path)

    recommender = TagBaseRecommender(
        mu=config.mu,
        neutral_rating=config.neutral_rating,
        use_pairwise_tags=config.use_pairwise_tags,
        use_item_priors=config.use_item_priors,
    )
    recommender.add_interactions(
        users=train_df[config.user_column].tolist(),
        items=train_df[config.item_column].tolist(),
        ratings=train_df[config.rating_column].apply(lambda x: normalize_overall_rating(x)).tolist(),
        tags=train_df[config.tag_column].apply(ast.literal_eval).tolist(),
        tag_ratings=None
    )
    recommender.fit()

    # Training/Items evaluation
    # https://dl.acm.org/doi/10.1145/2043932.2043996

    users = []
    selected_items = []
    item_relevances = []

    train_items = train_df[config.item_column].unique().tolist()
    grouped_train_df = train_df.groupby(config.user_column)
    grouped_test_df = test_df.groupby(config.user_column)

    for user_id in grouped_train_df.groups.keys():
        user_train_df = grouped_train_df.get_group(user_id)
        excluded_items = user_train_df[config.item_column].tolist()
        user_selected_items = set(train_items) - set(excluded_items)
        user_selected_items = list(user_selected_items)

        user_test_df = None
        user_item_relevances = {}
        if user_id in grouped_test_df.groups.keys():
            user_test_df = grouped_test_df.get_group(user_id)
            test_items = user_test_df[config.item_column].tolist()
            test_relevances = user_test_df[config.rating_column].apply(lambda x: max(0, x - 2)).tolist()
            user_item_relevances = {
                item: rating 
                for item, rating in zip(test_items, test_relevances) 
                if item in train_items
            }

        users.append(user_id)
        selected_items.append(user_selected_items)
        item_relevances.append(user_item_relevances)

    rankings = recommender.rank_all(users=users, items=selected_items)

    test_scores = evaluate_ranking_metrics(
        users=users,
        rankings=rankings,
        item_relevances=item_relevances,
        ks=[5, 10, 20],
    )
    print("Evaluation Results:")
    for metric, score in test_scores.items():
        print(f"{metric}: {score:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    config_path = f"{config.output_dir}/config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=4)

    res_path = f"{config.output_dir}/results.json"
    with open(res_path, "w") as f:
        json.dump(test_scores, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to the test data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output results.",
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Smoothing parameter for the recommender.",
    )
    parser.add_argument(
        "--neutral_rating",
        type=float,
        default=0.0,
        help="Neutral rating value for the recommender.",
    )
    parser.add_argument(
        "--use_pairwise_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use pairwise tags in the recommender.",
    )
    parser.add_argument(
        "--use_item_priors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use item priors in the recommender.",
    )

    parser.add_argument(
        "--user_column",
        type=str,
        default="user_id",
        help="Name of the user column in the data.",
    )
    parser.add_argument(
        "--item_column",
        type=str,
        default="item_id",
        help="Name of the item column in the data.",
    )
    parser.add_argument(
        "--rating_column",
        type=str,
        default="rating",
        help="Name of the rating column in the data.",
    )
    parser.add_argument(
        "--tag_column",
        type=str,
        default="aspects",
        help="Name of the tag column in the data.",
    )
    
    config = parser.parse_args()
    main(config)
