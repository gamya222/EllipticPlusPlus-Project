from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count


# --------------------------------------------------
# Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("InvestigateMissingFeatures")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# Read datasets
# --------------------------------------------------

features_path = "data/raw/elliptic/txs_features.csv"
classes_path = "data/raw/elliptic/txs_classes.csv"

features_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(features_path)
)

classes_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(classes_path)
)


# --------------------------------------------------
# The 18 features with NULL values
# --------------------------------------------------

missing_features = [
    "in_txs_degree",
    "out_txs_degree",
    "total_BTC",
    "fees",
    "size",
    "num_input_addresses",
    "num_output_addresses",
    "in_BTC_min",
    "in_BTC_max",
    "in_BTC_mean",
    "in_BTC_median",
    "in_BTC_total",
    "out_BTC_min",
    "out_BTC_max",
    "out_BTC_mean",
    "out_BTC_median",
    "out_BTC_total"
]


# ==================================================
# 1. CHECK IF THE SAME ROWS ARE MISSING
# ==================================================

print("\n========== MISSING FEATURE ROWS ==========")

any_missing_condition = None

for feature in missing_features:

    condition = col(feature).isNull()

    if any_missing_condition is None:
        any_missing_condition = condition
    else:
        any_missing_condition = (
            any_missing_condition | condition
        )


rows_with_any_missing = (
    features_df
    .filter(any_missing_condition)
)

print(
    "Rows with at least one missing feature:",
    rows_with_any_missing.count()
)


# Check whether ALL 18 are missing together

all_missing_condition = None

for feature in missing_features:

    condition = col(feature).isNull()

    if all_missing_condition is None:
        all_missing_condition = condition
    else:
        all_missing_condition = (
            all_missing_condition & condition
        )


rows_with_all_missing = (
    features_df
    .filter(all_missing_condition)
)

print(
    "Rows with all 18 features missing:",
    rows_with_all_missing.count()
)


# ==================================================
# 2. TIME STEP DISTRIBUTION
# ==================================================

print("\n========== TIME STEPS OF MISSING ROWS ==========")

rows_with_all_missing.groupBy(
    "Time step"
).count().orderBy(
    "Time step"
).show(50)


# ==================================================
# 3. JOIN WITH CLASS LABELS
# ==================================================

print("\n========== CLASSES OF MISSING ROWS ==========")

missing_with_class = (
    rows_with_all_missing
    .join(
        classes_df,
        on="txId",
        how="left"
    )
)

missing_with_class.groupBy(
    "class"
).count().orderBy(
    "class"
).show()


# ==================================================
# 4. SHOW SAMPLE TRANSACTIONS
# ==================================================

print("\n========== SAMPLE MISSING TRANSACTIONS ==========")

missing_with_class.select(
    "txId",
    "Time step",
    "class"
).show(20, truncate=False)


# ==================================================
# FINISH
# ==================================================

print("\n========== INVESTIGATION COMPLETE ==========")

spark.stop()