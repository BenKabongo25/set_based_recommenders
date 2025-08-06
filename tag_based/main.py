# Ben Kabongo
# August 2025


import argparse
import ast
import pandas as pd

from recommender import TagBaseRecommender
from utils import *


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

    users = []
    candidate_items = []

    grouped_test_df = test_df.groupby(config.user_column)
    for user_id, user_df in grouped_test_df:
        items = user_df[config.item_column].tolist()
        ratings = user_df[config.rating_column].apply(lambda x: normalize_overall_rating(x)).tolist()
        items_ratings = {item: rating for item, rating in zip(items, ratings)}

        users.append(user_id)
        candidate_items.append(items_ratings)

    rankings = recommender.rank_all(
        users=users,
        items=[list(candidate_items[i].keys()) for i in range(len(candidate_items))],
    )

    MAP = calculate_map(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
    )
    print(f"Mean Average Precision (MAP): {MAP:.4f}")

    MRR = calculate_mrr(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
    )
    print(f"Mean Reciprocal Rank (MRR): {MRR:.4f}")

    NDCG_at_5 = calculate_ndcg(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
        k=5,
    )
    print(f"NDCG@5: {NDCG_at_5:.4f}")

    NDCG_at_10 = calculate_ndcg(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
        k=10,
    )
    print(f"NDCG@10: {NDCG_at_10:.4f}")

    NDCG_at_20 = calculate_ndcg(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
        k=20,
    )
    print(f"NDCG@20: {NDCG_at_20:.4f}")

    HR_at_5 = hit_ratio_at_k(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
        k=5,
    )
    print(f"Hit Rate@5: {HR_at_5:.4f}")

    HR_at_10 = hit_ratio_at_k(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
        k=10,
    )
    print(f"Hit Rate@10: {HR_at_10:.4f}")

    HR_at_20 = hit_ratio_at_k(
        users=users,
        rankings=rankings,
        candidate_items=candidate_items,
        neutral_rating=config.neutral_rating,
        k=20,
    )
    print(f"Hit Rate@20: {HR_at_20:.4f}")


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
