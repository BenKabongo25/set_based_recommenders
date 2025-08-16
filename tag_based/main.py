# Ben Kabongo
# August 2025


import argparse
import ast
import json
import os
import pandas as pd

from recommender import TagBaseRecommender
from utils import evaluate_ranking_metrics, normalize_overall_rating, prepare_full_ranking_data


def main(config):
    train_df = pd.read_csv(config.train_data_path)
    test_df = pd.read_csv(config.test_data_path)
    train_df[config.tag_column] = train_df[config.tag_column].apply(ast.literal_eval)

    if config.use_clustered_tags:
        save_dir = os.path.join(config.output_dir, "w_clusters")
            
        all_tags = list(json.load(open(config.all_tags_path, 'r'))) # {"t": freq}
        selected_tags_index = json.load(open(config.selected_tags_index_path, 'r')) # ["t1", "t2", ...]
        tag2cluster_index = json.load(open(config.tag2cluster_index_path, 'r')) # {"t1": c1, t2: c2, ...}

        def process_tags(tag_set):
            cluster_set = set()
            for tag_index in tag_set:
                if tag_index not in selected_tags_index:
                    continue
                tag = all_tags[tag_index]
                if tag in tag2cluster_index:
                    cluster_set.add(tag2cluster_index[tag])
            return list(cluster_set)

        train_df[config.tag_column] = train_df[config.tag_column].apply(process_tags)

    else:
        save_dir = os.path.join(config.output_dir, "wo_clusters")

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
        tags=train_df[config.tag_column].tolist(),
        tag_ratings=None
    )
    recommender.fit()

    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=4)

    print("Full ranking evaluation:")
    users, selected_items, item_relevances = prepare_full_ranking_data(
        config=config,
        train_df=train_df,
        test_df=test_df,
        eval_df=None,  # No evaluation data provided
    )

    rankings = recommender.rank_all(users=users, items=selected_items)
    test_scores = evaluate_ranking_metrics(
        users=users,
        rankings=rankings,
        item_relevances=item_relevances,
        ks=[5, 10, 20, 50, 100],
    )
    for metric, score in test_scores.items():
        print(f"{metric}: {score:.6f}")

    full_res_path = os.path.join(save_dir, "results_full.json")
    with open(full_res_path, "w") as f:
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
        "--use_clustered_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use clustered tags in the recommender.",
    )
    parser.add_argument(
        "--all_tags_path",
        type=str,
        default=None,
        help="Path to the file containing all tags.",
    )
    parser.add_argument(
        "--selected_tags_index_path",
        type=str,
        default=None,
        help="Path to the file containing selected tags index.",
    )
    parser.add_argument(
        "--tag2cluster_index_path",
        type=str,
        default=None,
        help="Path to the file containing tag to cluster index mapping.",
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
        "--n_negative_per_positive",
        type=int,
        default=99,
        help="Number of negative items per positive item for sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
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
