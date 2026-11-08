# reciprocal-rec

An experimental two-sided recommender for reciprocal matches on LibimSeTi. It retrieves candidates with a PyTorch two-tower model and FAISS HNSW, then ranks them with a shared LightGBM classifier applied in both directions.

```text
LibimSeTi ZIP -> Spark preparation -> two-tower training -> cross-fitted LightGBM
                                                            |
User -> HNSW top 200 -> directed scores -> reciprocal fusion and penalty -> top 10
```

On the known-label test pool, reciprocal product improved NDCG@10 from 0.57517 to 0.57613 versus one-way ranking and recorded one fewer Mutual-Hit@10. The Docker service completed the local sequential benchmark at 4.94 ms HTTP p50 and 9.94 ms p99.

## Reproduce the offline experiment

Use Python 3.12 and Java 17. On Windows, run the offline pipeline in WSL. Obtain the original archive from the data source below, then run from the repository root.

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[data,retrieval,ranking,analysis]'
python -m reciprec.run --archive /path/to/libimseti-complete.zip
```

The command imports the ZIP, prepares Parquet, trains both models, and generates the baseline, retrieval, ranking, and exposure reports and plot. The default run directory is `results/run`. It contains the inputs, prepared data, model artifacts, configuration snapshot, and a `results/` subdirectory for the reports. Use `--output` to choose another fresh directory. `make all ARCHIVE=/path/to/libimseti-complete.zip` runs the same command.

To serve that run, set `RECIPREC_RUN_DIR=./results/run` in your shell and use the Compose instructions below. Point the host benchmark at `--embeddings results/run/artifacts/retrieval/embeddings.npz`.

GitHub Actions runs Ruff, pytest, and a synthetic archive-to-API smoke test on pull requests and pushes to `main`. The workflow uses generated ratings and checks that the service loads and uses the artifacts produced by the one-command run.

## Data and preprocessing

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
python -m pytest tests/test_retrieval.py
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

## Reciprocal ranking

After running two-tower retrieval, train one LightGBM binary classifier and apply it to each pair in both directions. Ranking inputs include training-derived user statistics, directional tower scores, user-statistic differences, and the gender combination.

```sh
python -m pip install -e '.[data,retrieval,ranking,test]'
python -m reciprec.ranking.ranker --device cpu
python -m pytest
```

The ranker uses two seeded folds of unordered pairs inside the training split. Both directions stay together. Each fold's scores and user statistics come from the other fold, so the ranker cannot use features computed from its own training labels for the same pair. Validation and test features use the full-training catalog and tower vectors. Each query's tower scores are standardized against its catalog score distribution to align the scales of independently trained towers. This preserves the two-tower baseline ordering within each query. The validation AUC selects the LightGBM iteration. [LightGBM training API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html)

The report compares six methods on the same test pool of reciprocally rated pairs used for the popularity baseline. This pool includes all observed gender combinations. The product and harmonic mean combine the two directional model outputs. The product is a ranking score and does not model the joint probability explicitly. The exposure penalty divides the product by `(training_likes_received + 1)^0.3`. The report also gives directed AUC on all rated test edges and separately for F→M and M→F.

The command writes `results/ranking.json` and saves the fitted booster in `artifacts/ranking/ranker.txt`. For another run, choose fresh `--artifacts` and `--report` paths, and use the retrieval artifacts produced from the same prepared data.

### LibimSeTi ranking results

The CPU run used 4,750,906 cross-fitted training examples. The best validation AUC was 0.91567 at the 150-round budget. Test AUC was 0.91541 overall, 0.91304 for F→M, and 0.91238 for M→F.

| Method | Mutual-Hit@10 | NDCG@10 | Exposure Gini | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 58.69% | 0.56826 | 0.89704 | 18.83% |
| Two-tower | 58.68% | 0.56798 | 0.89647 | 18.88% |
| One-way LightGBM | 58.69% | 0.57517 | 0.89484 | 19.03% |
| Reciprocal harmonic mean | 58.68% | 0.57610 | 0.89528 | 18.98% |
| Reciprocal product | 58.68% | 0.57613 | 0.89534 | 18.97% |
| Reciprocal product with exposure penalty | 58.69% | 0.57524 | 0.89217 | 19.25% |

The product increased NDCG@10 by 0.00096 over the one-way ranker and yielded one fewer hit among the 9,960 evaluated users. The short candidate lists leave little headroom in Mutual-Hit@10. At the fixed penalty of 0.3, exposure Gini decreased from 0.89534 to 0.89217 and NDCG@10 decreased to 0.57524.

## Exposure penalty analysis

```sh
python -m pip install -e '.[retrieval,ranking,analysis]'
python -m reciprec.eval.exposure
```

The sweep reuses the saved ranker and the same known-label test pool. It writes `results/exposure.json` and `results/exposure_tradeoff.png`. The table describes sensitivity on this test pool.

| α | Mutual-Hit@10 | NDCG@10 | Exposure Gini | Coverage |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 58.6847% | 0.57613 | 0.89534 | 18.97% |
| 0.1 | 58.6847% | 0.57587 | 0.89326 | 19.14% |
| 0.2 | 58.6847% | 0.57555 | 0.89270 | 19.19% |
| 0.3 | 58.6948% | 0.57524 | 0.89217 | 19.25% |
| 0.4 | 58.6948% | 0.57432 | 0.89167 | 19.30% |
| 0.5 | 58.6948% | 0.57363 | 0.89125 | 19.35% |

Increasing α from 0 to 0.5 lowered exposure Gini by 0.00409 and increased coverage by 0.38 percentage points. NDCG@10 fell by 0.00251. The difference in Mutual-Hit@10 amounts to one additional user with a hit.

Query cohorts use quartile cutoffs of training likes received across the full user catalog, keeping ties together. The counts below cover the 9,960 evaluated queries.

| Training likes received | Queries | NDCG change, product vs. one-way | NDCG change, α=0.3 vs. product |
| --- | ---: | ---: | ---: |
| 0–2 | 962 | 0.00025 | 0.00014 |
| 3–7 | 1,426 | 0.00186 | 0.00031 |
| 8–19 | 2,322 | 0.00107 | -0.00077 |
| 20–∞ | 5,250 | 0.00080 | -0.00146 |

The 3–7 group had the largest NDCG gain from reciprocal scoring. At α=0.3, the 20-or-more group had the largest drop in NDCG relative to the unpenalized product. These differences describe the observed candidate pool.

## Recommendation API

After preparing the data and training both models, start the local service with Docker Compose. The container mounts the prepared data and model artifacts read-only and publishes the API on the host loopback interface.

```sh
docker compose up --build -d --wait
curl http://127.0.0.1:8000/health
curl 'http://127.0.0.1:8000/recommend/-1?k=10&preferred_gender=F'
docker compose down
```

Use a retained user ID in `GET /recommend/{user_id}?k=10&alpha=0.3` for personalized results. It searches the opposite-gender index, excludes candidates previously rated in training, then uses the shared LightGBM scorer and reciprocal product with an exposure penalty. Responses include both directional predictions, the final score, and stage timings. They can contain fewer than `k` items when fewer eligible candidates are available. An unknown ID returns penalized popularity with `fallback: true` and null directional predictions. The optional `preferred_gender` filters that fallback, which otherwise uses the full catalog.

With the container running, execute the benchmark from the host. It samples 1,000 distinct known users with seed 42 and makes sequential requests over a persistent HTTP client after 20 warm-up requests.

```sh
python -m pip install numpy httpx
python -m reciprec.serving.benchmark
```

The report is written to `results/serving.json`. This Windows host to Docker Desktop run used one Uvicorn worker and one thread each for FAISS and LightGBM predictions. The HTTP measurement includes the round trip and response transfer.

| Stage | p50 (ms) | p99 (ms) |
| --- | ---: | ---: |
| http | 4.94 | 9.94 |
| retrieval | 0.54 | 1.49 |
| features | 0.20 | 0.44 |
| ranking | 1.79 | 2.50 |

For local service tests, install `.[serving,http-test,test]` and run `python -m pytest tests/test_serving.py`.

## Design walkthrough

- **Why split by user pair?** Each pair has up to two directed ratings. Grouping them in both the outer split and the ranking folds keeps a held-out rating and its reverse together. Each ranking training row uses statistics and tower scores generated from the opposite fold.
- **Why two towers?** A user's outgoing preferences and incoming appeal are different roles. Candidate vectors can be computed once, indexed, and retrieved cheaply by inner product. Compare HNSW with exact search to separate index approximation from model recall.
- **Why two directional predictions?** The ranker consumes ordered pairs. Swapping the users changes their feature roles and yields a separate estimate of the reverse direction. The product rewards candidates with strong scores in both directions.
- **What can the experiment tell us?** Ranking metrics use only reciprocally rated held-out pairs, and most evaluated queries have at most ten such candidates. The archive lacks timestamps and preference settings, so evaluation uses a pair split and retrieval assumes opposite-gender eligibility. The service numbers describe sequential requests on one local worker. The experiment provides offline ranking and exposure evidence.
