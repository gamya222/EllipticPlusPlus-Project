from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# --------------------------------------------------
# Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("EllipticTemporalSplit")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# File paths
# --------------------------------------------------

features_path = "data/raw/elliptic/txs_features.csv"
classes_path = "data/raw/elliptic/txs_classes.csv"


# --------------------------------------------------
# Read feature data
# --------------------------------------------------

print("\n========== READING FEATURE DATA ==========")

features_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(features_path)
)


# --------------------------------------------------
# Read class data
# --------------------------------------------------

print("\n========== READING CLASS DATA ==========")

classes_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(classes_path)
)


# --------------------------------------------------
# Join features with labels
# --------------------------------------------------

print("\n========== JOINING FEATURES + LABELS ==========")

transactions_df = (
    features_df
    .join(
        classes_df,
        on="txId",
        how="left"
    )
)

print("Joined transaction count:", transactions_df.count())


# --------------------------------------------------
# Temporal split
# --------------------------------------------------

print("\n========== TEMPORAL SPLIT ==========")

train_df = transactions_df.filter(
    col("Time step") <= 34
)

test_df = transactions_df.filter(
    col("Time step") >= 35
)


# --------------------------------------------------
# Count rows
# --------------------------------------------------

print("\n========== SPLIT SIZES ==========")

print("Training transactions:", train_df.count())
print("Testing transactions:", test_df.count())


# --------------------------------------------------
# Time ranges
# --------------------------------------------------

print("\n========== TIME RANGES ==========")

print("Training time steps:")
train_df.select("Time step").distinct().orderBy(
    "Time step"
).show(50)

print("Testing time steps:")
test_df.select("Time step").distinct().orderBy(
    "Time step"
).show(50)


# --------------------------------------------------
# Class distribution
# --------------------------------------------------

print("\n========== TRAIN CLASS DISTRIBUTION ==========")

train_df.groupBy(
    "class"
).count().orderBy(
    "class"
).show()


print("\n========== TEST CLASS DISTRIBUTION ==========")

test_df.groupBy(
    "class"
).count().orderBy(
    "class"
).show()


# --------------------------------------------------
# Finish
# --------------------------------------------------

print("\n========== TEMPORAL SPLIT COMPLETE ==========")

spark.stop()