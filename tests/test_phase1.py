from math import log2
from zipfile import ZipFile

import pytest
from pyspark.sql import SparkSession, functions as F

from reciprec.data.download import extract_inputs
from reciprec.data.prepare import assign_splits, directed_core, eligible_ratings, user_features
from reciprec.eval.popularity import evaluate


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    session = (
        SparkSession.builder.master("local[2]").appName("reciprec-test")
        .config("spark.sql.shuffle.partitions", "2").getOrCreate()
    )
    session.sparkContext.setCheckpointDir(tmp_path_factory.mktemp("checkpoints").as_uri())
    yield session
    session.stop()


def test_extract_inputs(tmp_path):
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as data:
        data.writestr("nested/ratings.dat", "1,2,9\n")
        data.writestr("nested/gender.dat", "1,F\n2,M\n")
    output = tmp_path / "raw"
    extract_inputs(archive, output)
    assert (output / "ratings.dat").read_text() == "1,2,9\n"
    assert (output / "gender.dat").read_text() == "1,F\n2,M\n"
    with pytest.raises(FileExistsError):
        extract_inputs(archive, output)


def test_core_peels_until_stable(spark):
    ratings = spark.createDataFrame(
        [(1, 2, 9), (2, 1, 8), (2, 3, 7), (3, 4, 6), (1, 1, 10), (4, 5, 10)],
        "src INT, dst INT, rating INT",
    )
    users = spark.createDataFrame([(i,) for i in range(1, 5)], "user_id INT")
    core = directed_core(eligible_ratings(ratings, users), 1, 1)
    try:
        assert {(r.src, r.dst) for r in core.collect()} == {(1, 2), (2, 1)}
    finally:
        core.unpersist()


def test_split_and_training_features(spark):
    ratings = spark.createDataFrame(
        [(1, other, 9, 1) for other in range(2, 52)]
        + [(other, 1, 2, 0) for other in range(2, 52)],
        "src INT, dst INT, rating INT, liked INT",
    )
    assigned = assign_splits(ratings, 0.8, 0.1, 0.1, 42)
    pair_splits = assigned.groupBy(F.greatest("src", "dst")).agg(F.countDistinct("split").alias("n"))
    assert all(row.n == 1 for row in pair_splits.collect())
    counts = {row.split: row["count"] for row in assigned.groupBy("split").count().collect()}
    assert set(counts) == {"train", "val", "test"}
    train = assigned.where(F.col("split") == "train")
    users = spark.createDataFrame([(1, "F"), (100, "M")], "user_id INT, gender STRING")
    active, cold = user_features(train, users).orderBy("user_id").collect()
    assert active.ratings_given == active.ratings_received == counts["train"] // 2
    assert active.mean_rating_given == 9
    assert active.likes_received == 0
    assert cold.ratings_given == 0
    assert cold.mean_rating_given is None


def test_popularity_metrics(spark):
    test = spark.createDataFrame(
        [(1, 2, 1), (2, 1, 1), (1, 3, 0), (3, 1, 1), (1, 4, 1)],
        "src INT, dst INT, liked INT",
    )
    features = spark.createDataFrame(
        [(1, 1), (2, 2), (3, 3), (4, 4)], "user_id INT, likes_received INT"
    )
    result = evaluate(test, features, k=1)
    assert result["evaluated_users"] == 3
    assert result["reciprocal_pairs"] == 2
    assert result["mutual_like_pairs"] == 1
    assert result["mutual_hit@1"] == pytest.approx(1 / 3)
    assert result["ndcg@1"] == pytest.approx(1 / 3)
    assert result["coverage"] == 0.5
    assert result["exposure_gini"] == pytest.approx(7 / 12)

    result = evaluate(test, features, k=2)
    assert result["mutual_hit@2"] == pytest.approx(2 / 3)
    assert result["ndcg@2"] == pytest.approx((1 + 1 / log2(3)) / 3)
