# reciprocal-rec

An experimental two-sided recommender for reciprocal matches. Given a user A, the system will retrieve candidates B, score both A's interest in B and B's interest in A, and rank candidates by their combined scores.

## First milestone

Prepare LibimSeTi ratings locally, split by unordered user pair, compute user features from the training split, and report a popularity baseline on held-out pairs with ratings in both directions.

| Path | Purpose |
| --- | --- |
| `configs/default.yaml` | Initial preprocessing and split settings |
| `reciprec/data/` | Archive import, directed k-core filtering, pair split, and train-only features |
| `reciprec/eval/` | Popularity baseline and offline metrics |

### Run locally

Use Python 3.11 or 3.12 and a JDK supported by [PySpark 3.5.8](https://spark.apache.org/docs/3.5.8/api/python/getting_started/install.html), such as Java 17. Set `JAVA_HOME` to the JDK. On Windows, run in WSL with Python and Java installed inside WSL, or use native Hadoop binaries compatible with the Hadoop version bundled with Spark.

Download the archived copy of the original LibimSeTi ZIP. It contains the headerless `ratings.dat` (`UserID,ProfileID,Rating`) and `gender.dat` (`UserID,Gender`) files in a shared user ID space. [Archived original ZIP](https://web.archive.org/web/20151130002651id_/http://www.occamslab.com/petricek/data/libimseti-complete.zip)

The archive used for the baseline has SHA-256 `83a2f3fcb80c2c6af8fe286044dffca27d886507461cef4dd5b1b885522a2577`. Its 17,359,346 ratings reference 135,359 raters and 168,791 rated profiles, all present among the 220,970 user IDs in `gender.dat`.

Keep the data local for research use and obtain permission before redistributing it. Cite Brožovský and Petříček (2007) when publishing results. [Original study](https://arxiv.org/abs/cs/0703042) · [Archived dataset page and usage terms](https://web.archive.org/web/20200901085122/http://www.occamslab.com/petricek/data/)

```sh
python -m pip install -e '.[data,test]'
python -m reciprec.data.download --archive /path/to/libimseticomplete.zip
python -m reciprec.data.prepare
python -m reciprec.eval.popularity
python -m pytest tests/test_phase1.py
```

For an authorized HTTPS source, pass `--url` in place of `--archive`. If you already have both files under `data/raw/`, start with `prepare`. The importer and preprocessor expect fresh output directories. To rerun preprocessing with different settings, pass a new `--output` to `prepare` and point `popularity --prepared` to it.

Preprocessing excludes self-ratings, users without known gender, and out-of-range ratings. Duplicate directed ratings retain the highest score. It writes `train`, `val`, `test`, and `user_features` Parquet datasets under `data/prepared/`. The fixed-seed hash split targets 80/10/10 by unordered pair because the source lacks timestamps. `data/prepared/summary.json` records the retained score histogram and positive rate.

The directed k-core with at least 10 ratings given and 10 received retains 49,780 users and 5,938,610 ratings. The resulting train, validation, and test splits contain 4,750,906, 594,354, and 593,350 directed ratings. A score of 9 or 10 is a like. This cutoff labels 28.79% of training ratings positive, compared with 38.62% for a cutoff of 8 and 22.12% for a cutoff of 10.

### LibimSeTi baseline

The baseline writes `results/popularity.json`. It ranks candidates by training likes received, breaking ties by candidate ID. Candidates come from test pairs with ratings in both directions. Mutual-Hit@10 and NDCG@10 average over users with at least one such candidate, including users with no mutual likes as zeros. Coverage and exposure Gini use all retained users, including those who receive no exposure.

| Method | Mutual-Hit@10 | NDCG@10 | Exposure Gini | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 58.69% | 0.5683 | 0.8970 | 18.83% |

The test pool has 11,571 reciprocally rated pairs across 9,960 users, including 7,295 mutual likes. The median user has one candidate, and 97.63% of evaluated users have at most 10. The maximum achievable Mutual-Hit@10 in this pool is 58.73%, the share of users with at least one mutual like. Popularity is 0.04 percentage points below that ceiling, so this metric leaves little room to separate rankers under the known-label candidate protocol.

## Two-tower retrieval

The source and candidate towers each combine a user ID embedding with six training-only user statistics to produce a 64-dimensional vector. Training uses positive edges from `train`. Other targets in each batch serve as implicit negatives. Repeated target IDs share a logit, and logQ subtracts the log of each target's estimated batch inclusion probability to account for frequent targets appearing in more batches.

After preprocessing, install the retrieval dependencies and run training. The first command selects a CPU-only PyTorch wheel for the WSL setup. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[data,retrieval,test]'
python -m reciprec.retrieval.train --device cpu
python -m pytest
```

Validation uses exact inner-product retrieval on a fixed sample of eligible users to select the best checkpoint. Test evaluation uses HNSW over the full eligible candidate catalog, with separate indexes for `F` and `M`. The dataset lacks preference settings, so this experiment assumes opposite-gender eligibility. Recall@200 averages the per-user fraction of eligible held-out likes retrieved, over users with at least one such like. Self matches and candidates the user rated in training are excluded from search. An audit compares HNSW with exact top-200 for the same vectors, queries, and eligibility rules. [FAISS HNSW index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#indexhnsw-variants)

Direct inner-product HNSW had poor agreement with exact search on the learned vectors. The index adds one coordinate to each candidate to make all squared norms equal within its partition, and adds zero to each query. Squared Euclidean distance then differs from negative twice the original dot product by a constant for each query. The model vectors remain 64-dimensional, and the graph stores 65-dimensional vectors. Search expands when filtering leaves too few candidates and falls back to exact scoring if it reaches the full partition. [FAISS notes on mapping inner product to L2](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-do-max-inner-product-search-on-indexes-that-support-only-l2)

The command writes `results/retrieval.json` and saves the model weights, both embedding tables, and FAISS indexes under `artifacts/retrieval/`. For another run, choose fresh `--artifacts` and `--report` paths.

### LibimSeTi retrieval results

The CPU run used 1,367,731 positive training edges. Epoch 8 had the best validation Recall@200 within the eight-epoch budget.

| Evaluation | Queries | Result |
| --- | ---: | ---: |
| Validation Recall@200, exact search | 2,000 | 52.44% |
| Test Recall@200, HNSW | 35,034 | 51.89% |
| HNSW overlap with exact top 200, test audit | 1,000 | 99.28% |
