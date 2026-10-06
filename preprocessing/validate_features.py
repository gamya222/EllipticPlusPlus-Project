from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, isnan, sum as spark_sum


# --------------------------------------------------
# Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("EllipticFeatureValidation")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# Read feature dataset
# --------------------------------------------------

features_path = "data/raw/elliptic/txs_features.csv"

features_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(features_path)
)


# ==================================================
# 1. BASIC INFORMATION
# ==================================================

print("\n========== FEATURE DATASET ==========")

print("Rows:", features_df.count())
print("Columns:", len(features_df.columns))


# ==================================================
# 2. TXID VALIDATION
# ==================================================

print("\n========== TXID VALIDATION ==========")

null_txids = features_df.filter(
    col("txId").isNull()
).count()

print("Null txId:", null_txids)


duplicate_txids = (
    features_df
    .groupBy("txId")
    .count()
    .filter(col("count") > 1)
    .count()
)

print("Duplicate txId values:", duplicate_txids)


# ==================================================
# 3. TIME STEP VALIDATION
# ==================================================

print("\n========== TIME STEP VALIDATION ==========")

null_timesteps = features_df.filter(
    col("Time step").isNull()
).count()

print("Null Time step:", null_timesteps)


time_stats = features_df.select(
    min("Time step").alias("min_time"),
    max("Time step").alias("max_time")
).collect()[0]

print("Minimum Time step:", time_stats["min_time"])
print("Maximum Time step:", time_stats["max_time"])


invalid_timesteps = features_df.filter(
    (col("Time step") < 1) |
    (col("Time step") > 49)
).count()

print("Invalid Time step values:", invalid_timesteps)


# ==================================================
# 4. FEATURE VALUE VALIDATION
# ==================================================

print("\n========== FEATURE VALUE VALIDATION ==========")

# Remove metadata columns
feature_columns = [
    c for c in features_df.columns
    if c not in ["txId", "Time step"]
]

print("Number of feature columns:", len(feature_columns))


# Build one Spark aggregation
null_nan_exprs = []

for c in feature_columns:

    null_nan_exprs.append(
        spark_sum(
            col(c).isNull().cast("int")
        ).alias(f"{c}_null")
    )

    null_nan_exprs.append(
        spark_sum(
            isnan(col(c)).cast("int")
        ).alias(f"{c}_nan")
    )


quality_row = features_df.select(
    null_nan_exprs
).collect()[0]


# Display problematic features only
problem_found = False

for c in feature_columns:

    null_count = quality_row[f"{c}_null"]
    nan_count = quality_row[f"{c}_nan"]

    if null_count > 0 or nan_count > 0:

        problem_found = True

        print(
            f"{c}: "
            f"NULL = {null_count}, "
            f"NaN = {nan_count}"
        )


if not problem_found:
    print("No NULL or NaN values found in feature columns.")


# ==================================================
# FINISH
# ==================================================

print("\n========== FEATURE VALIDATION COMPLETE ==========")

spark.stop()