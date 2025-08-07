# Ben Kabongo
# August 2025


import numpy as np
from tqdm import tqdm
from typing import Dict, List, Union

from item_model import ItemModel
from user_model import UserModel


class TagBaseRecommender:
    """
    Tag-based Recommender.

    Attributes:
        user_models (Dict[Union[str, int], UserModel]): Dictionary mapping user IDs to UserModel instances.
        item_models (Dict[Union[str, int], ItemModel]): Dictionary mapping item IDs to ItemModel instances.
        n_users (int): Number of users in the recommender system.
        n_items (int): Number of items in the recommender system.
        mu (float): Regularization parameter for tag scores.
        neutral_rating (float): Neutral rating value used for initialization.
        use_pairwise_tags (bool): Whether to use pairwise tags in scoring.
        use_item_priors (bool): Whether to use item priors in scoring.
        tag_items (Dict[Union[str, int], set]): Dictionary mapping tag IDs to sets of item IDs rated with that tag.
        n_tag_items (Dict[Union[str, int], int]): Dictionary mapping tag IDs to the number of items rated with that tag.
        n_pairwise_tag_items (Dict[tuple, int]): Dictionary mapping pairs of tag IDs to the number of items rated with both tags.

    Functions:
        add_interaction(user_id, item_id, rating, tags=None, tag_ratings=None):
            Add an interaction (user-item rating) to the recommender system.
        add_interactions(users, items, ratings, tags=None, tag_ratings=None):
            Add multiple interactions to the recommender system.
        fit():
            Fit the recommender system by computing user and item models.
        rank(user_id, items):
            Rank items for a given user based on the learned models.
        rank_all(users, items):
            Rank items for multiple users.
    """

    def __init__(
        self,
        mu: float = 0.0,
        neutral_rating: float = 0.0,
        use_pairwise_tags: bool = True,
        use_item_priors: bool = True,
    ) -> None:
        self.user_models = {}
        self.item_models = {}

        self.n_users = 0
        self.n_items = 0

        self.mu = mu
        self.neutral_rating = neutral_rating
        self.use_pairwise_tags = use_pairwise_tags
        self.use_item_priors = use_item_priors

        self.tag_items = {} # tag_id: set of item_ids rated with this tag
        self.n_tag_items = {} # tag_id: number of items rated with this tag
        self.n_pairwise_tag_items = {}

    def add_interaction(
        self,
        user_id: Union[str, int],
        item_id: Union[str, int],
        rating: float,
        tags: List[Union[str, int]] = None,
        tag_ratings: Union[None, List[float]] = None
    ) -> None:
        """
        Add an interaction to the recommender system.

        Args:
            user_id (Union[str, int]): The ID of the user.
            item_id (Union[str, int]): The ID of the item.
            rating (float): The rating given by the user to the item.
            tags (List[Union[str, int]], optional): List of tags associated with the item. Defaults to None.
            tag_ratings (Union[None, List[float]], optional): Ratings for each tag. Defaults to None.
                If provided, it should match the length of `tags`.
                If `None`, the rating for the item will be used for all tags. (as in the original paper)
        """
        if user_id not in self.user_models:
            self.user_models[user_id] = UserModel(user_id, self.neutral_rating, self.use_pairwise_tags)
        if item_id not in self.item_models:
            self.item_models[item_id] = ItemModel(item_id, self.neutral_rating, self.use_pairwise_tags)

        self.user_models[user_id].add_interaction(item_id, rating, tags, tag_ratings)
        self.item_models[item_id].add_interaction(user_id, rating, tags, tag_ratings)

        if tags is None:
            return
        
        for tag in tags:
            if tag not in self.tag_items:
                self.tag_items[tag] = set()
            self.tag_items[tag].add(item_id)

    def add_interactions(
        self,
        users: List[Union[str, int]],
        items: List[Union[str, int]],
        ratings: List[float],
        tags: List[List[Union[str, int]]] = None,
        tag_ratings: Union[None, List[List[float]]] = None
    ) -> None:
        """
        Add multiple interactions to the recommender system.

        Args:
            users (List[Union[str, int]]): List of user IDs.
            items (List[Union[str, int]]): List of item IDs.
            ratings (List[float]): List of ratings corresponding to each user-item pair.
            tags (List[List[Union[str, int]]], optional): List of lists of tags for each item. Defaults to None.
            tag_ratings (Union[None, List[List[float]]], optional): List of lists of tag ratings for each item. Defaults to None.
        """
        for user_id, item_id, rating, item_tags, item_tag_ratings in zip(users, items, ratings, tags or [], tag_ratings or []):
            self.add_interaction(user_id, item_id, rating, item_tags, item_tag_ratings)

    def fit(self) -> None:
        """
        Fit the recommender system by computing user and item models.
        """
        for user_model in tqdm(
            self.user_models.values(), 
            desc="Fitting user models", total=len(self.user_models)):
            user_model.fit()

        for item_model in tqdm(
            self.item_models.values(), 
            desc="Fitting item models", total=len(self.item_models)):
            item_model.fit()

        self.n_users = len(self.user_models)
        self.n_items = len(self.item_models)

        for tag_a, tag_items_a in self.tag_items.items():
            self.n_tag_items[tag_a] = len(tag_items_a)

            if not self.use_pairwise_tags:
                continue

            for tag_b, tag_items_b in self.tag_items.items():
                if tag_a == tag_b:
                    continue

                common_tag_items = tag_items_a.intersection(tag_items_b)
                n_common_tag_items = len(common_tag_items)
                if n_common_tag_items > 0:
                    self.n_pairwise_tag_items[(tag_a, tag_b)] = n_common_tag_items

    def _compute_set_tag_score(
        self,
        tags: List[Union[str, int]],
        user_tag_weights: Dict[Union[str, int], float],
        item_tag_weights: Dict[Union[str, int], float],
        n_tag_items: Dict[Union[str, int], int],
        mu: float = 0.0,
        n_items: int = 0
    ):
        """
        Compute the score for a set of tags based on user and item tag weights.

        Args:
            tags (List[Union[str, int]]): List of tag IDs.
            user_tag_weights (Dict[Union[str, int], float]): User tag weights.
            item_tag_weights (Dict[Union[str, int], float]): Item tag weights.
            n_tag_items (Dict[Union[str, int], int]): Number of items rated with each tag.
            mu (float): Regularization parameter.
            n_items (int): Total number of items.

        Returns:
            float: Computed score for the set of tags.
        """
        score = 0.0
        for tag_id in tags:
            if tag_id not in item_tag_weights:
                continue

            user_tag_weight = user_tag_weights.get(tag_id, 0.0)
            item_tag_weight = item_tag_weights[tag_id]
            n_tag_items_count = n_tag_items.get(tag_id, 0)

            num = item_tag_weight + mu * (n_tag_items_count / n_items)
            denom = mu + n_items / n_tag_items_count
            tag_score = np.log(num/denom)
            
            score += user_tag_weight * tag_score

        return score

    def rank(
        self,
        user_id: Union[str, int],
        items: List[Union[str, int]]
    ) -> List[Union[str, int]]:
        """
        Rank items for a given user based on the learned models.

        Args:
            user_id (Union[str, int]): The ID of the user.
            items (List[Union[str, int]]): List of item IDs to rank.

        Returns:
            List[Union[str, int]]: Ranked list of item IDs.
        """
        user_model = self.user_models.get(user_id, UserModel(user_id, self.neutral_rating))
        item_scores = {}

        for item_id in items:
            item_model = self.item_models.get(item_id, ItemModel(item_id, self.neutral_rating))

            score = 0.0
            if self.use_item_priors:
                score += np.log(item_model.n_pos_ratings + 1) - np.log(self.n_users + 1 - item_model.n_neg_ratings)

            score += self._compute_set_tag_score(
                tags=list(user_model.tag_weights.keys()),
                user_tag_weights=user_model.tag_weights,
                item_tag_weights=item_model.tag_weights,
                n_tag_items=self.n_tag_items,
                mu=self.mu,
                n_items=self.n_items
            )

            if self.use_pairwise_tags:
                score += self._compute_set_tag_score(
                    tags=list(user_model.pairwise_tag_weights.keys()),
                    user_tag_weights=user_model.pairwise_tag_weights,
                    item_tag_weights=item_model.pairwise_tag_weights,
                    n_tag_items=self.n_pairwise_tag_items,
                    mu=self.mu,
                    n_items=self.n_items
                )

            item_scores[item_id] = score

        ranked_items = sorted(item_scores, key=item_scores.get, reverse=True)
        return ranked_items
    
    def rank_all(
        self,
        users: List[Union[str, int]],
        items: List[List[Union[str, int]]]
    ) -> List[List[Union[str, int]]]:
        """
        Rank items for multiple users.

        Args:
            users (List[Union[str, int]]): List of user IDs.
            items (List[List[Union[str, int]]]): List of lists of item IDs for each user.

        Returns:
            List[List[Union[str, int]]]: Ranked list of item IDs for each user.
        """
        ranked_items = []
        for user_id, user_items in tqdm(
            zip(users, items), 
            desc="Ranking", total=len(users)):
            ranked_items.append(self.rank(user_id, user_items))
        return ranked_items
