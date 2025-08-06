# Ben Kabongo
# August 2025


import numpy as np
from typing import List, Union


class ItemModel:
    """
    ItemModel represents an item in the Tag-Based Recommender System.

    Attributes:
        item_id (Union[str, int]): The ID of the item.
        neutral_rating (float): The neutral rating value.
        use_pairwise_tags (bool): Whether to use pairwise tags in the model.
        user_ratings (dict): Dictionary mapping user IDs to ratings given by the user to the item.
        tag_interaction_ratings (dict): Dictionary mapping tag IDs to another dictionary of user IDs and their ratings.
        n_pos_ratings (int): Number of positive ratings for the item.
        n_neg_ratings (int): Number of negative ratings for the item.
        tag_weights (dict): Dictionary mapping tag IDs to their weights.
        pairwise_tag_weights (dict): Dictionary mapping pairs of tag IDs to their pairwise weights.

    Functions:
        add_interaction(
        user_id: Union[str, int], rating: float, tags: List[Union[str, int]] = None, tag_ratings: Union[None, List[float]] = None) -> None:
            Add an interaction for the item with a user, including optional tags and tag ratings.
        fit() -> None:
            Fit the item model by computing tag weights and pairwise tag weights.
    """

    def __init__(
        self,
        item_id: Union[str, int],
        neutral_rating: float = 0.0,
        use_pairwise_tags: bool = True,
    ) -> None:
        self.item_id = item_id
        self.neutral_rating = neutral_rating
        self.use_pairwise_tags = use_pairwise_tags
        
        self.user_ratings = {}  # user_id: rating
        self.tag_interaction_ratings = {}  # tag_id: {user_id: tag/item rating}

        self.n_pos_ratings = 0
        self.n_neg_ratings = 0

        self.tag_weights          = {}  # tag_id: x
        self.pairwise_tag_weights = {}  # (tag_id_1, tag_id_2): x

    def add_interaction(
        self,
        user_id: Union[str, int],
        rating: float,
        tags: List[Union[str, int]] = None,
        tag_ratings: Union[None, List[float]] = None,
    ) -> None:
        """
        Add an interaction for the item.

        Args:
            user_id (Union[str, int]): The ID of the user.
            rating (float): The rating given by the user to the item.
            tags (List[Union[str, int]], optional): List of tags associated with the item. Defaults to None.
            tag_ratings (Union[None, List[float]], optional): Ratings for each tag. Defaults to None.
                If provided, it should match the length of `tags`.
                If `None`, the rating for the item will be used for all tags. (as in the original paper)
        """
        self.user_ratings[user_id] = rating
        
        if tags is not None:
            if tag_ratings is None:
                tag_ratings = [rating] * len(tags)
            assert len(tags) == len(tag_ratings), "Tags and tag ratings must have the same length."

            for tag_id, tag_rating in zip(tags, tag_ratings):
                if tag_id not in self.tag_interaction_ratings:
                    self.tag_interaction_ratings[tag_id] = {}
                self.tag_interaction_ratings[tag_id][user_id] = tag_rating

    def fit(self) -> None:
        """
        Fit the item model by computing tag weights and pairwise tag weights.
        """
        ratings = np.ndarray(list(self.user_ratings.values()))
        self.n_pos_ratings = np.sum(ratings >= self.neutral_rating)
        self.n_neg_ratings = np.sum(ratings < self.neutral_rating)

        for tag_a, tag_interaction_ratings_a in self.tag_interaction_ratings.items():
            self.tag_weights[tag_a] = 1

            if not self.use_pairwise_tags:
                continue

            for tag_b, tag_interaction_ratings_b in self.tag_interaction_ratings.items():
                if tag_a == tag_b:
                    continue

                common_tag_users = set(tag_interaction_ratings_a.keys()).intersection(set(tag_interaction_ratings_b.keys()))
                n_common_tag_users = len(common_tag_users)
                if n_common_tag_users == 0:
                    continue
                self.pairwise_tag_weights[(tag_a, tag_b)] = 1
