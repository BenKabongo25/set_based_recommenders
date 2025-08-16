# Ben Kabongo
# August 2025


import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple, Union


def normalize_overall_rating(
    rating: float, 
    min_rating: float=1, 
    max_rating: float=5,
    new_min_rating: float=-1,
    new_max_rating: float=1
) -> float:
    """
    Normalize the overall rating to a new scale.

    Args:
        rating (float): The original rating to normalize.
        min_rating (float): The minimum value of the original rating scale.
        max_rating (float): The maximum value of the original rating scale.
        new_min_rating (float): The minimum value of the new rating scale.
        new_max_rating (float): The maximum value of the new rating scale.

    Returns:
        float: The normalized rating on the new scale.
    """
    return ((rating - min_rating) / (max_rating - min_rating)) * (new_max_rating - new_min_rating) + new_min_rating


def prepare_full_ranking_data(
    config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[Union[str, int]], List[List[Union[str, int]]], List[Dict[Union[str, int], float]]]:
    """
    Prepares data for full ranking evaluation.
        
    Args:
        config: Configuration object with user_column, item_column, rating_column
        train_df: Training dataframe
        test_df: Test dataframe
        eval_df: Evaluation dataframe (optional, defaults to None)
            
    Returns:
        Tuple containing:
                - users: List of users
                - selected_items: List of selectable items for each user
                - item_relevances: List of relevances for each user
    """
    users = []
    selected_items = []
    item_relevances = []

    include_eval = eval_df is not None
    
    train_items = train_df[config.item_column].unique().tolist()
    grouped_train_df = train_df.groupby(config.user_column)
    grouped_test_df = test_df.groupby(config.user_column)
    if include_eval:
        grouped_eval_df = eval_df.groupby(config.user_column)
    
    for user_id in grouped_train_df.groups.keys():
        user_train_df = grouped_train_df.get_group(user_id)
        
        excluded_items = user_train_df[config.item_column].tolist()
        if include_eval:
            user_eval_df = grouped_eval_df.get_group(user_id)
            excluded_items += user_eval_df[config.item_column].tolist()
        
        user_selected_items = set(train_items) - set(excluded_items)
        user_selected_items = list(user_selected_items)
        
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
    
    return users, selected_items, item_relevances


def extract_positive_items(
    users: List[Union[str, int]],
    item_relevances: List[Dict[Union[str, int], float]],
    min_relevance: float = 1.0
) -> Dict[Union[str, int], List[Union[str, int]]]:
    """
    Extracts the positive items for each user.
        
    Args:
        users: List of users
        item_relevances: List of relevances for each user
        min_relevance: Minimum threshold to consider an item as positive. Defaults to 1.0.
            
    Returns:
        Dictionary {user_id: [dict of positive items]}
    """
    positive_items = {}
    
    for user, relevances in zip(users, item_relevances):
        positive_items[user] = {
            item: relevance for item, relevance in relevances.items() 
            if relevance >= min_relevance
        }
    
    return positive_items


def create_uniform_sampling_pairs(
    users: List[Union[str, int]],
    selected_items: List[List[Union[str, int]]],
    positive_items: Dict[Union[str, int], Dict[Union[str, int], float]],
    n_negative_per_positive: int = 5,
    seed: Optional[int] = None
) -> Tuple[List[Union[str, int]], List[List[Union[str, int]]], List[Dict[Union[str, int], float]]]:
    """
    Creates positive/negative item pairs by uniform sampling.
        
    Args:
        users: List of users
        selected_items: List of selectable items for each user
        positive_items: Dictionary of positive items per user
        n_negative_per_positive: Number of negative items per positive item
        seed: Seed for reproducibility
            
    Returns:
        Tuple containing:
            - sampled_users: List of users (repeated according to number of pairs)
            - sampled_items: List of items sampled for each pair
            - sampled_relevances: List of relevances for each pair
    """
    if seed is not None:
        random.seed(seed)
    
    sampled_users = []
    sampled_items = []
    sampled_relevances = []
    
    for user, user_selected_items in zip(users, selected_items):
        user_positive_items = positive_items.get(user, {})
        negative_candidates = list(set(user_selected_items) - set(user_positive_items))
        
        for positive_item in user_positive_items:
            if len(negative_candidates) >= n_negative_per_positive:
                sampled_negatives = random.sample(negative_candidates, n_negative_per_positive)
            else:
                sampled_negatives = negative_candidates.copy()
            
            pair_items = [positive_item] + sampled_negatives
            
            pair_relevances = {positive_item: user_positive_items[positive_item]}
            for neg_item in sampled_negatives:
                pair_relevances[neg_item] = 0.0
            
            sampled_users.append(user)
            sampled_items.append(pair_items)
            sampled_relevances.append(pair_relevances)
    
    return sampled_users, sampled_items, sampled_relevances


def prepare_ranking_data(
    config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    n_negative_per_positive: int = 99,
    min_relevance: float = 0.5,
    seed: Optional[int] = None,
    full_ranking: bool = True,
    sample_ranking: bool = True,
) -> Dict[str, float]:
    """
    Prepares data for ranking evaluation, either full or sampled.

    Args:
        config: Configuration object with user_column, item_column, rating_column
        train_df: Training dataframe
        test_df: Test dataframe
        eval_df: Evaluation dataframe (optional)
        n_negative_per_positive: Number of negative items per positive item for sampling
        min_relevance: Minimum relevance threshold for positive items
        seed: Seed for reproducibility
        full_ranking: Whether to prepare full ranking data
        sample_ranking: Whether to prepare sampled ranking data

    Returns:
        Dictionary containing:
            - "full": Full ranking data (if full_ranking is True)
            - "sampled": Sampled ranking data (if sample_ranking is True)
    """
    ret = {}

    users, selected_items, item_relevances = prepare_full_ranking_data(
        config, train_df, test_df, eval_df,
    )
    if full_ranking:
        ret["full"] = {
            "users": users,
            "selected_items": selected_items,
            "item_relevances": item_relevances,
        }

    if not sample_ranking:
        return ret

    positive_items = extract_positive_items(
        users, item_relevances, min_relevance=min_relevance
    )
    
    sampled_users, sampled_items, sampled_relevances = create_uniform_sampling_pairs(
        users, selected_items, positive_items, 
        n_negative_per_positive=n_negative_per_positive,
        seed=seed
    )

    ret["sampled"] = {
        "users": sampled_users,
        "selected_items": sampled_items,
        "item_relevances": sampled_relevances,
    }

    return ret


def evaluate_ranking_metrics(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]],
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate a set of ranking metrics: NDCG@k, Hit@k, Precision@k, and Recall@k.

    Args:
        users: List of user IDs.
        rankings: Ranked items for each user.
        item_relevances: Relevance scores per user.
        ks: List of cutoff ranks for top-k metrics.

    Returns:
        Dictionary of metric names and their values.
    """
    results = {}

    for k in ks:
        recall = recall_at_k(users, rankings, item_relevances, k)
        precision = precision_at_k(users, rankings, item_relevances, k)
        ndcg = calculate_ndcg(users, rankings, item_relevances, k)
        hit = hit_ratio_at_k(users, rankings, item_relevances, k)

        results[f'Recall@{k}'] = recall
        results[f'Precision@{k}'] = precision
        results[f'NDCG@{k}'] = ndcg
        results[f'Hit Rate@{k}'] = hit

    return results


def dcg_at_k(relevances: List[float], k: int) -> float:
    relevances = np.asarray(relevances)
    if len(relevances) == 0:
        return 0.0
    if len(relevances) > k:
        relevances = relevances[:k]
    return np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))


def ndcg_at_k_single_user(
    ranked_items: List[Union[str, int]],
    item_relevance: Dict[Union[str, int], float],
    k: int = 10,
) -> float:
    """
    Compute NDCG@k for a single user.
    Args:
        ranked_items: Ranked list of items.
        item_relevance: Dictionary of item -> relevance score.
        k: Truncation level.
    Returns:
        float: NDCG@k score.
    """
    true_relevances = [item_relevance.get(item, 0.0) for item in ranked_items]
    dcg = dcg_at_k(true_relevances, k)
    
    sorted_relevances = sorted(item_relevance.values(), reverse=True)
    idcg = dcg_at_k(sorted_relevances, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_ndcg(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]],
    k: int = 10,
) -> float:
    """
    Compute mean NDCG@k over all users.

    Args:
        users: List of user IDs.
        rankings: List of ranked items per user.
        item_relevances: List of relevance score dictionaries per user.
        k: Truncation level.

    Returns:
        float: Mean NDCG@k.
    """
    ndcgs = []

    for u in range(len(users)):
        ndcg = ndcg_at_k_single_user(rankings[u], item_relevances[u], k)
        ndcgs.append(ndcg)

    return np.mean(ndcgs)


def hit_ratio_at_k(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]],
    k: int = 10,
) -> float:
    """
    Compute Hit Ratio@k (Hit@k): whether there's at least one relevant item in top-k.

    Args:
        users: List of user IDs.
        rankings: Ranked item lists per user.
        item_relevances: Relevance scores per user.
        k: Number of top items to consider.

    Returns:
        float: Mean Hit Ratio@k.
    """
    hits = []

    for u in range(len(users)):
        rankings_u = rankings[u]
        u_k = min(k, len(rankings_u))
        top_k_items = rankings[u][:u_k]
        user_relevances = item_relevances[u]

        hit = any(user_relevances.get(item, 0) > 0 for item in top_k_items)
        hits.append(1.0 if hit else 0.0)

    return np.mean(hits)


def precision_at_k(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]],
    k: int = 10,
) -> float:
    """
    Compute mean Precision@k over a list of users.

    Args:
        users: List of user IDs.
        rankings: List of ranked item lists for each user.
        item_relevances: List of dictionaries mapping item IDs to relevance scores.
        k: Number of top items to consider.

    Returns:
        float: Mean Precision@k.
    """
    precisions = []

    for u in range(len(users)):
        rankings_u = rankings[u]
        u_k = min(k, len(rankings_u))
        top_k_items = rankings_u[:u_k]
        user_relevances = item_relevances[u]

        relevant_count = sum(
            user_relevances.get(item, 0) > 0 for item in top_k_items
        )
        precision = relevant_count / u_k if u_k > 0 else 0.0
        precisions.append(precision)

    return np.mean(precisions)


def recall_at_k(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]],
    k: int = 10,
) -> float:
    """
    Compute Recall@k: proportion of relevant items found in top-k.

    Args:
        users: List of user IDs.
        rankings: Ranked item lists per user.
        item_relevances: Relevance score dicts per user.
        k: Truncation level.

    Returns:
        float: Mean Recall@k.
    """
    recalls = []

    for u in range(len(users)):
        rankings_u = rankings[u]
        u_k = min(k, len(rankings_u))
        top_k_items = rankings_u[:u_k]
        user_relevances = item_relevances[u]

        # Set of relevant items for this user
        relevant_items = {item for item, rel in user_relevances.items() if rel > 0}

        if not relevant_items:
            recalls.append(0.0)
            continue

        retrieved_relevant = sum(1 for item in top_k_items if item in relevant_items)
        recall = retrieved_relevant / len(relevant_items)
        recalls.append(recall)

    return np.mean(recalls)
