"""Evaluate a train-only popularity baseline on reciprocally rated test pairs."""

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F

from reciprec.eval.metrics import exposure_gini


def reciprocal_candidates(test):
    reverse = test.select(
        F.col("dst").alias("src"), F.col("src").alias("dst"), F.col("liked").alias("reverse_liked")
    )
    return test.join(reverse, ["src", "dst"]).select(
        "src", "dst", (F.col("liked") * F.col("reverse_liked")).alias("mutual_like")
    )


def evaluate(test, features, k=10):
    if k < 1:
        raise ValueError("k must be positive")
    popularity = features.select(F.col("user_id").alias("dst"), "likes_received")
    candidates = reciprocal_candidates(test).join(popularity, "dst", "left").fillna(0, ["likes_received"])
    order = Window.partitionBy("src").orderBy(F.desc("likes_received"), "dst")
    ranked = candidates.withColumn("rank", F.row_number().over(order)).persist()
    try:
        per_user = ranked.groupBy("src").agg(
            F.count("*").alias("candidate_count"),
            F.sum("mutual_like").alias("positives"),
            F.max(F.when(F.col("rank") <= k, F.col("mutual_like")).otherwise(0)).alias("hit"),
            F.sum(F.when(F.col("rank") <= k, F.col("mutual_like") / F.log2(F.col("rank") + 1))
                  .otherwise(0.0)).alias("dcg"),
        )
        ideal_dcg = F.lit(0.0)
        for position in range(1, k + 1):
            ideal_dcg = ideal_dcg + F.when(
                F.col("positives") >= position, 1 / F.log2(F.lit(position + 1))
            ).otherwise(0.0)
        summary = per_user.agg(
            F.count("*").alias("evaluated_users"),
            (F.sum("candidate_count") / 2).cast("long").alias("reciprocal_pairs"),
            (F.sum("positives") / 2).cast("long").alias("mutual_like_pairs"),
            F.avg("hit").alias(f"mutual_hit@{k}"),
            F.avg(F.when(F.col("positives") > 0, F.col("dcg") / ideal_dcg)
                  .otherwise(0.0)).alias(f"ndcg@{k}"),
        ).first().asDict()
        if not summary["evaluated_users"]:
            raise ValueError("No reciprocally rated test pairs; baseline metrics are undefined")
        exposure = ranked.where(F.col("rank") <= k).groupBy("dst").count()
        histogram = {
            row.exposures: row.users
            for row in popularity.join(exposure, "dst", "left").fillna(0, ["count"])
            .groupBy(F.col("count").alias("exposures")).agg(F.count("*").alias("users")).collect()
        }
    finally:
        ranked.unpersist()
    population = sum(histogram.values())
    return {
        "method": "popularity",
        **summary,
        "exposure_gini": exposure_gini(histogram),
        "coverage": (population - histogram.get(0, 0)) / population,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("results/popularity.json"))
    args = parser.parse_args()
    spark = (
        SparkSession.builder.master("local[2]").appName("reciprec-popularity")
        .config("spark.sql.shuffle.partitions", "16").getOrCreate()
    )
    try:
        test = spark.read.parquet(str((args.prepared / "test").resolve()))
        features = spark.read.parquet(str((args.prepared / "user_features").resolve()))
        result = evaluate(test, features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
