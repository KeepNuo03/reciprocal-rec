# reciprocal-rec

An experimental two-sided recommender for reciprocal matches. Given a user A, the system will retrieve candidates B, score both A's interest in B and B's interest in A, and rank candidates by their combined scores.

## First milestone

Prepare LibimSeTi ratings locally, split by unordered user pair, compute user features from the training split, and report a popularity baseline on held-out pairs with ratings in both directions. The raw dataset and generated artifacts belong in local directories excluded from Git.

| Path | Purpose |
| --- | --- |
| `configs/default.yaml` | Initial preprocessing and split settings |
| `reciprec/data/` | Rating preprocessing and train-only features |
| `reciprec/eval/` | Baselines and offline metrics |

The settings are provisional until the rating distribution and the number of reciprocal pairs have been measured. Dataset ingestion, preprocessing, and baseline evaluation are pending.
