# Ben Kabongo
# August 2025


import numpy as np
from typing import Any, Dict, List, Union


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


def evaluate_ranking_metrics(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]],
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate a set of ranking metrics: MAP, MRR, NDCG@k, Hit@k.

    Args:
        users: List of user IDs.
        rankings: Ranked items for each user.
        item_relevances: Relevance scores per user.
        ks: List of cutoff ranks for top-k metrics.

    Returns:
        Dictionary of metric names and their values.
    """
    results = {}

    # Global metrics
    results['MAP'] = calculate_map(users, rankings, item_relevances)
    results['MRR'] = calculate_mrr(users, rankings, item_relevances)

    # Top-k metrics
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


def average_precision(
    ranked_items: List[Union[str, int]],
    item_relevance: Dict[Union[str, int], float]
) -> float:
    """
    Compute Average Precision (AP) for a single user.

    Args:
        ranked_items: List of ranked item IDs.
        item_relevance: Dictionary mapping item IDs to relevance scores.

    Returns:
        float: Average Precision.
    """
    num_relevant = 0
    precisions = []

    for i, item in enumerate(ranked_items, start=1):
        if item_relevance.get(item, 0) > 0:
            num_relevant += 1
            precisions.append(num_relevant / i)

    if num_relevant == 0:
        return 0.0

    return np.mean(precisions)


def calculate_map(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]]
) -> float:
    """
    Compute Mean Average Precision (MAP) over users.

    Args:
        users: List of user IDs.
        rankings: List of ranked item lists per user.
        item_relevances: List of dictionaries mapping item IDs to relevance scores.

    Returns:
        float: MAP score.
    """
    ap_scores = []

    for u in range(len(users)):
        ap = average_precision(rankings[u], item_relevances[u])
        ap_scores.append(ap)

    return np.mean(ap_scores)


def dcg_at_k(relevances: List[float], k: int) -> float:
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
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
    true_relevances = [item_relevance.get(item, 0) > 0 for item in ranked_items]
    dcg = dcg_at_k(true_relevances, k)

    sorted_relevances = sorted(item_relevance.values(), reverse=True)
    ideal_relevances = [rel > 0 for rel in sorted_relevances]
    idcg = dcg_at_k(ideal_relevances, k)

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


def calculate_mrr(
    users: List[Union[str, int]],
    rankings: List[List[Union[str, int]]],
    item_relevances: List[Dict[Union[str, int], float]]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) over all users.

    Args:
        users: List of user IDs.
        rankings: Ranked item lists per user.
        item_relevances: Relevance score dicts per user.

    Returns:
        float: MRR score.
    """
    reciprocal_ranks = []

    for u in range(len(users)):
        for rank, item in enumerate(rankings[u], start=1):
            if item_relevances[u].get(item, 0) > 0:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


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
        top_k_items = rankings[u][:k]
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
        top_k_items = rankings[u][:k]
        user_relevances = item_relevances[u]

        relevant_count = sum(
            user_relevances.get(item, 0) > 0 for item in top_k_items
        )
        precision = relevant_count / k
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
        top_k_items = rankings[u][:k]
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
