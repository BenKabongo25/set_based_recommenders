# Ben Kabongo
# August 2025


import numpy as np
from typing import Dict, List, Tuple, Union


class UserModel:
    """
    UserModel represents a user in the Tag-Based Recommender System.

    Attributes:
        user_id (Union[str, int]): The ID of the user.
        neutral_rating (float): The neutral rating value.
        use_pairwise_tags (bool): Whether to use pairwise tags in the model.
        item_ratings (dict): Dictionary mapping item IDs to ratings given by the user.
        tag_interaction_ratings (dict): Dictionary mapping tag IDs to another dictionary of item IDs and their ratings.
        tag_weights (dict): Dictionary mapping tag IDs to their weights.
        tag_ratings (dict): Dictionary mapping tag IDs to their average ratings.
        tag_coverages (dict): Dictionary mapping tag IDs to their coverage values.
        tag_significances (dict): Dictionary mapping tag IDs to their significance values.
        tag_utilies (dict): Dictionary mapping tag IDs to their utility values.
        pairwise_tag_weights (dict): Dictionary mapping pairs of tag IDs to their pairwise weights.
        pairwise_tag_ratings (dict): Dictionary mapping pairs of tag IDs to their pairwise ratings.
        pairwise_tag_coverages (dict): Dictionary mapping pairs of tag IDs to their pairwise coverages.
        pairwise_tag_significances (dict): Dictionary mapping pairs of tag IDs to their pairwise significances.
        pairwise_tag_utilies (dict): Dictionary mapping pairs of tag IDs to their pairwise utilities.

    Functions:
        add_interaction(
        item_id: Union[str, int], rating: float, tags: List[Union[str, int]] = None, tag_ratings: Union[None, List[float]] = None) -> None:
            Add an interaction for the user with an item, including optional tags and tag ratings.
        fit() -> None:
            Fit the user model by computing tag ratings, weights, coverages, significances, and utilities.
    """

    def __init__(
        self,
        user_id: Union[str, int],
        neutral_rating: float = 0.0,
        use_pairwise_tags: bool = True,
    ) -> None:
        self.user_id = user_id
        self.neutral_rating = neutral_rating
        self.use_pairwise_tags = use_pairwise_tags
        
        self.item_ratings = {} # item_id: rating
        self.tag_interaction_ratings = {}  # tag_id: {item_id: tag/item rating}

        self.tag_weights        = {} # tag_id: x
        self.tag_ratings        = {}
        self.tag_coverages      = {}
        self.tag_significances  = {}
        self.tag_utilies        = {}
        self.tag_n_items        = {}

        self.pairwise_tag_weights       = {}  # (tag_id_1, tag_id_2): x
        self.pairwise_tag_ratings       = {}
        self.pairwise_tag_coverages     = {}
        self.pairwise_tag_significances = {}
        self.pairwise_tag_utilies       = {}
        self.pairwise_tag_n_items       = {}

    def add_interaction(
        self,
        item_id: Union[str, int],
        rating: float,
        tags: List[Union[str, int]] = None,
        tag_ratings: Union[None, List[float]] = None,
    ) -> None:
        """
        Add an interaction for the user.

        Args:
            item_id (Union[str, int]): The ID of the item.
            rating (float): The rating given by the user to the item.
            tags (List[Union[str, int]], optional): List of tags associated with the item. Defaults to None.
            tag_ratings (Union[None, List[float]], optional): Ratings for each tag. Defaults to None.
                If provided, it should match the length of `tags`.
                If `None`, the rating for the item will be used for all tags. (as in the original paper)
        """
        self.item_ratings[item_id] = rating
        
        if tags is not None:
            if tag_ratings is None:
                tag_ratings = [rating] * len(tags)
            assert len(tags) == len(tag_ratings), "Tags and tag ratings must have the same length."

            for tag_id, tag_rating in zip(tags, tag_ratings):
                if tag_id not in self.tag_interaction_ratings:
                    self.tag_interaction_ratings[tag_id] = {}
                self.tag_interaction_ratings[tag_id][item_id] = tag_rating

    def _compute_coverage(
        self,
        n_tags: int,
        n_total: int
    ) -> float:
        """
        Compute the coverage of a tag based on the number of items rated with that tag.

        Args:
            n_tags (int): Number of items rated with the tag.
            n_total (int): Total number of rated items.

        Returns:
            float: Coverage value, which is the minimum of the ratio of rated items to total items
                   and the ratio of non-rated items to total items.
        """
        return min(n_tags / n_total, (n_total - n_tags) / n_total)
    
    def _compute_significance(
        self,
        weight: float,
        std: float,
        n_items: int
    ) -> float:
        """
        Compute the significance of a tag based on its weight, standard deviation, and number of items.

        Args:
            weight (float): The weight of the tag.
            std (float): The standard deviation of the tag ratings.
            n_items (int): Number of items rated with the tag.

        Returns:
            float: Significance value, which is the minimum of 2 and the ratio of absolute weight to
                   the adjusted standard deviation.
        """
        return min(2, np.abs(weight) / (std / (np.sqrt(n_items) + 1e-12) + 1e-12))

    def fit(self) -> None:
        """
        Fit the user model by computing tag ratings, weights, coverages, significances, and utilities.
        """
        n_rated_items = len(self.item_ratings)

        for tag_a, tag_interaction_ratings_a in self.tag_interaction_ratings.items():
            n_tag_a_items = len(tag_interaction_ratings_a)

            tag_ratings_a = np.array(list(tag_interaction_ratings_a.values()))
            tag_rating_a = np.mean(tag_ratings_a)
            tag_std_a = np.std(tag_ratings_a)
            tag_weight_a = tag_rating_a - self.neutral_rating
            tag_coverage_a = self._compute_coverage(n_tag_a_items, n_rated_items)
            tag_significance_a = self._compute_significance(tag_weight_a, tag_std_a, n_tag_a_items)
            tag_utility_a = tag_coverage_a * tag_significance_a * np.abs(tag_weight_a)

            self.tag_ratings[tag_a] = tag_rating_a
            self.tag_weights[tag_a] = tag_weight_a
            self.tag_coverages[tag_a] = tag_coverage_a
            self.tag_significances[tag_a] = tag_significance_a
            self.tag_utilies[tag_a] = tag_utility_a
            self.tag_n_items[tag_a] = n_tag_a_items

            if not self.use_pairwise_tags:
                continue

            for tag_b, tag_interaction_ratings_b in self.tag_interaction_ratings.items():
                if tag_a == tag_b:
                    continue

                common_tag_items = set(tag_interaction_ratings_a.keys()).intersection(set(tag_interaction_ratings_b.keys()))
                n_common_tag_items = len(common_tag_items)
                if n_common_tag_items == 0:
                    continue

                filtered_ratings_a = np.array([tag_interaction_ratings_a[item_id] for item_id in common_tag_items])
                filtered_ratings_b = np.array([tag_interaction_ratings_b[item_id] for item_id in common_tag_items])
                pairwise_ratings = np.array((filtered_ratings_a + filtered_ratings_b) / 2)

                pairwise_rating = np.mean(pairwise_ratings)
                pairwise_std = np.std(pairwise_ratings)
                pairwise_weight = pairwise_rating - tag_rating_a
                pairwise_coverage = self._compute_coverage(n_common_tag_items, n_tag_a_items)
                pairwise_significance = self._compute_significance(pairwise_weight, pairwise_std, n_common_tag_items)
                pairwise_utility = pairwise_coverage * pairwise_significance * np.abs(pairwise_weight)

                self.pairwise_tag_ratings[(tag_a, tag_b)] = pairwise_rating
                self.pairwise_tag_weights[(tag_a, tag_b)] = pairwise_weight
                self.pairwise_tag_coverages[(tag_a, tag_b)] = pairwise_coverage
                self.pairwise_tag_significances[(tag_a, tag_b)] = pairwise_significance
                self.pairwise_tag_utilies[(tag_a, tag_b)] = pairwise_utility
                self.pairwise_tag_n_items[(tag_a, tag_b)] = n_common_tag_items

    def _compute_conditional_utility(
        self,
        tag: Union[str, int],
        weights: Dict[Union[str, int], float],
        significances: Dict[Union[str, int], float],
        n_items: Dict[Union[str, int], int],
        n_selected_items: int
    ) -> float:
        """
        Compute the conditional utility of a tag based on its weight, significance, and number of items.

        Args:
            tag (Union[str, int]): The ID of the tag.
            weights (Dict[Union[str, int], float]): Dictionary mapping tag IDs to their weights.
            significances (Dict[Union[str, int], float]): Dictionary mapping tag IDs to their significances.
            n_items (Dict[Union[str, int], int]): Dictionary mapping tag IDs to the number of items rated with that tag.
            n_selected_items (int): Number of items for selected tags.

        Returns:
            float: Conditional utility value for the specified tag.
        """
        tag_weight = weights[tag]
        tag_significance = significances[tag]
        n_tag_items = n_items[tag]
        tag_coverage = self._compute_coverage(n_tag_items / n_selected_items, len(self.item_ratings))
        tag_utility = tag_coverage * tag_significance * np.abs(tag_weight)
        return tag_utility
    
    def topk_tags(
        self,
        k: int = 10
    ) -> List[Union[str, int]]:
        selected_tags = set()
        unique_selected_tags = set()

        tag_utilities = self.tag_utilies.copy()
        if self.use_pairwise_tags:
            tag_utilities.update(self.pairwise_tag_utilies)

        sorted_tags = sorted(tag_utilities.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_tags) > 0:
            tag = sorted_tags[0][0]
            selected_tags.add(tag)
            if type(tag) is tuple:
                unique_selected_tags.update(tag)
            else:
                unique_selected_tags.add(tag)

        n_items = len(self.item_ratings)
        for _ in range(k - 1):
            selected_tag_items = set()
            for tag in unique_selected_tags:
                selected_tag_items.update(self.tag_interaction_ratings.get(tag, {}).keys())
            n_selected_tag_items = len(selected_tag_items)

            utilities = {}
            for tag in self.tag_weights:
                if tag in selected_tags:
                    continue
                utilities[tag] = self._compute_conditional_utility(
                    tag,
                    self.tag_weights,
                    self.tag_significances,
                    self.tag_n_items,
                    n_selected_tag_items
                )

            if not self.use_pairwise_tags:
                continue

            for tag in self.pairwise_tag_weights:
                if tag in selected_tags:
                    continue
                utilities[tag] = self._compute_conditional_utility(
                    tag,
                    self.pairwise_tag_weights,
                    self.pairwise_tag_significances,
                    self.pairwise_tag_n_items,
                    n_selected_tag_items
                )

            if len(utilities) == 0:
                break

            sorted_utilities = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
            tag = sorted_utilities[0][0]
            selected_tags.add(tag)
            if type(tag) is tuple:
                unique_selected_tags.update(tag)
            else:
                unique_selected_tags.add(tag)

        return list(selected_tags)
    