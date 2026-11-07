"""Prepare LibimSeTi ratings for leakage-safe model training."""

import argparse
import json
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from pyspark.sql import SparkSession, functions as F


def read_inputs(spark, raw):
    ratings = spark.read.schema("src INT, dst INT, rating INT").csv(
        str(raw / "ratings.dat"), mode="FAILFAST"
    )
    users = spark.read.schema("user_id INT, gender STRING").csv(
        str(raw / "gender.dat"), mode="FAILFAST"
    )
    users = users.where(F.col("gender").isin("F", "M"))
    return ratings, users


def eligible_ratings(ratings, users):
    ids = users.select("user_id")
    return (
        ratings.where((F.col("src") != F.col("dst")) & F.col("rating").between(1, 10))
        .join(ids.withColumnRenamed("user_id", "src"), "src", "left_semi")
        .join(ids.withColumnRenamed("user_id", "dst"), "dst", "left_semi")
        .groupBy("src", "dst").agg(F.max("rating").alias("rating"))
    )


def directed_core(ratings, min_given, min_received):
    """Peel users until both degree thresholds hold in the remaining graph."""
    current = ratings.persist()
    size = current.count()
    while size:
        given = current.groupBy("src").count().where(F.col("count") >= min_given)
        received = current.groupBy("dst").count().where(F.col("count") >= min_received)
        keep = given.select(F.col("src").alias("user_id")).join(
            received.select(F.col("dst").alias("user_id")), "user_id"
        )
        filtered = current.join(
            keep.withColumnRenamed("user_id", "src"), "src", "left_semi"
        ).join(keep.withColumnRenamed("user_id", "dst"), "dst", "left_semi").persist()
        filtered_size = filtered.count()
        if filtered_size == size:
            filtered.unpersist()
            break
        # Bound the execution plan across peeling iterations.
        checkpoint = filtered.checkpoint(eager=True)
        filtered.unpersist()
        current.unpersist()
        current, size = checkpoint.persist(), filtered_size
    return current


def assign_splits(ratings, train, validation, test, seed):
    if min(train, validation, test) <= 0 or not isclose(train + validation + test, 1.0):
        raise ValueError("Split fractions must be positive and sum to 1")
    pair_lo = F.least("src", "dst")
    pair_hi = F.greatest("src", "dst")
    bucket = F.pmod(F.xxhash64(pair_lo, pair_hi, F.lit(seed)), F.lit(1_000_000))
    fraction = bucket / 1_000_000
    return ratings.withColumn(
        "split",
        F.when(fraction < train, "train").when(fraction < train + validation, "val")
        .otherwise("test"),
    )


def user_features(train, users):
    given = train.groupBy(F.col("src").alias("user_id")).agg(
        F.count("*").alias("ratings_given"),
        F.avg("rating").alias("mean_rating_given"),
        F.avg("liked").alias("like_rate_given"),
    )
    received = train.groupBy(F.col("dst").alias("user_id")).agg(
        F.count("*").alias("ratings_received"),
        F.avg("rating").alias("mean_rating_received"),
        F.avg("liked").alias("like_rate_received"),
        F.sum("liked").alias("likes_received"),
    )
    return users.join(given, "user_id", "left").join(received, "user_id", "left").fillna(
        0, subset=["ratings_given", "ratings_received", "likes_received"]
    )


def prepare(spark, raw, output, config):
    ratings, users = read_inputs(spark, raw)
    settings = config["preprocessing"]
    given = settings["min_ratings_given"]
    received = settings["min_ratings_received"]
    threshold = settings["like_threshold"]
    if min(given, received) < 1 or not 1 <= threshold <= 10:
        raise ValueError("Degree minima must be positive and like_threshold must be 1–10")
    core = directed_core(eligible_ratings(ratings, users), given, received)
    try:
        histogram = {row.rating: row["count"] for row in core.groupBy("rating").count().collect()}
        if not histogram:
            raise ValueError("No ratings remain after eligibility and degree filtering")
        labeled = core.withColumn("liked", (F.col("rating") >= threshold).cast("int"))
        splits = assign_splits(labeled, **config["split"])
        active_ids = core.select(F.col("src").alias("user_id")).distinct()
        train = splits.where(F.col("split") == "train").drop("split")
        for split in ("train", "val", "test"):
            edges = splits.where(F.col("split") == split).drop("split")
            edges.write.parquet(str(output / split))
        features = user_features(train, users.join(active_ids, "user_id", "left_semi"))
        features.write.parquet(str(output / "user_features"))
        summary = {
            "rating_histogram": histogram,
            "positive_rate": sum(n for rating, n in histogram.items() if rating >= threshold)
            / sum(histogram.values()),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        core.unpersist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Choose a fresh output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master("local[2]").appName("reciprec-prepare")
        .config("spark.sql.shuffle.partitions", "16").getOrCreate()
    )
    try:
        with TemporaryDirectory(prefix=".reciprec-", dir=output.parent) as directory:
            staging = Path(directory)
            spark.sparkContext.setCheckpointDir(str(staging / "checkpoints"))
            prepare(spark, args.raw.resolve(), staging / "prepared", config)
            (staging / "prepared").rename(output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
