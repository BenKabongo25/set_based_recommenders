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

    data_df = pd.concat([train_df, test_df], ignore_index=True)
    all_items = data_df[config.item_column].unique().tolist()

    users = []
    item_relevances = []

    grouped_df = data_df.groupby(config.user_column)
    for user_id, user_df in grouped_df:
        items = user_df[config.item_column].tolist()
        ratings = user_df[config.rating_column].apply(lambda x: normalize_overall_rating(x)).tolist()
        items_ratings = {item: rating for item, rating in zip(items, ratings)}

        users.append(user_id)
        item_relevances.append(items_ratings)

    rankings = recommender.rank_all(
        users=users,
        items=[all_items] * len(users),
    )

    test_scores = evaluate_ranking_metrics(
        users=users,
        rankings=rankings,
        item_relevances=item_relevances,
        neutral_rating=config.neutral_rating,
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
