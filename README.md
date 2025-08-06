# Transparent, Scrutable and Explainable User Models for Personalized Recommendation

This repository contains a faithful implementation of the model proposed in the SIGIR 2019 paper:

> [**Transparent, Scrutable and Explainable User Models for Personalized Recommendation**](https://doi.org/10.1145/3331184.3331211)
> *Krisztian Balog, Filip Radlinski, Shushan Arakelyan.*

## 📌 Overview

Traditional recommender systems struggle to be both *transparent* (showing how recommendations are computed) and *scrutable* (allowing users to correct their profile). This repository implements a **tag-based recommendation model** that satisfies:

* **Explainability:** User preferences are modeled as interpretable tag-level statements.
* **Transparency:** Recommendations are derived directly from the model structure.
* **Scrutability:** Users can revise the tags or tag-pair interactions they agree or disagree with, and see the immediate effect on recommendations.

## 🧠 Core Ideas from the Paper

* **Set-Based User Preferences:**
  Each user’s profile is represented as a weighted set of tags (and optionally tag pairs). We infer these weights from explicit ratings on items annotated with tags.

* **Pairwise Tag Interactions:**
  User preferences are enriched by modeling second-order interactions like "I like **action** movies **with humor**" or "I dislike **romance** unless it is **comedy**".

* **Transparent Scoring Function:**
  Each recommendation score is computed as a weighted combination of user-tag and item-tag interactions, optionally regularized with popularity priors.

## 🧪 Usage Example

```python
from recommender import TagBaseRecommender

recommender = TagBaseRecommender(mu=10.0, neutral_rating=0.0)

# Add user-item interactions
recommender.add_interaction(
    user_id="u1", 
    item_id="i1", 
    rating=1.0,
    tags=["action", "thriller"]
)
recommender.add_interaction(
    user_id="u1", 
    item_id="i2", 
    rating=-1.0,
    tags=["romance"]
)

# ... add interactions

# Fit models
recommender.fit()

# Rank candidate items
ranked = recommender.rank(user_id="u1", items=["i3", "i4", "i5"])
print("Top recommendations:", ranked)
```


| Parameter | Description | Default |
|-----------|-------------|---------|
| `mu` | Smoothing parameter for tag probability estimation | 0.0 |
| `neutral_rating` | Threshold for neutral ratings | 0.0 |
| `use_pairwise_tags` | Enable pairwise tag interactions | True |
| `use_item_priors` | Include popularity-based item priors | True |


## 🔍 Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{balog2019transparent,
  title={Transparent, Scrutable and Explainable User Models for Personalized Recommendation},
  author={Balog, Krisztian and Radlinski, Filip and Arakelyan, Shushan},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={265--274},
  year={2019}
}
```