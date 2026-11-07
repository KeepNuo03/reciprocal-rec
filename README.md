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

Obtain an authorized copy of the original LibimSeTi ZIP containing the headerless `ratings.dat` (`UserID,ProfileID,Rating`) and `gender.dat` (`UserID,Gender`) files. The historical download links were unavailable during development. The [Netzschleuder listing](https://networks.skewed.de/net/libimseti) documents the ratings graph and lists edge weights as its only property. See the [original study](https://arxiv.org/abs/cs/0703042) and the [dataset's usage terms and format](https://github.com/ChicagoBoothML/DATA___DatingAgency/blob/master/readme.txt). Keep the data local for research use and obtain permission before redistributing it.

```sh
python -m pip install -e '.[data,test]'
python -m reciprec.data.download --archive /path/to/libimseticomplete.zip
python -m reciprec.data.prepare
python -m reciprec.eval.popularity
python -m pytest
```

For an authorized HTTPS source, pass `--url` in place of `--archive`. If you already have both files under `data/raw/`, start with `prepare`. The importer and preprocessor expect fresh output directories. To rerun preprocessing with different settings, pass a new `--output` to `prepare` and point `popularity --prepared` to it.

Preprocessing excludes self-ratings, users without known gender, and out-of-range ratings. Duplicate directed ratings retain the highest score. It writes `train`, `val`, `test`, and `user_features` Parquet datasets under `data/prepared/`. The fixed-seed hash split targets 80/10/10 by unordered pair because the source lacks timestamps. `data/prepared/summary.json` records the retained score histogram and positive rate. The degree and like thresholds in `configs/default.yaml` are provisional.

The baseline writes `results/popularity.json`. It ranks candidates by training likes received, breaking ties by candidate ID. Candidates come from test pairs with ratings in both directions. Mutual-Hit@10 and NDCG@10 average over users with at least one such candidate, including users with no mutual likes as zeros. Coverage and exposure Gini use all retained users, including those who receive no exposure. This protocol measures re-ranking of known-label candidates. Real-data metrics are pending.
