from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# --------------------------------------------------
# Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SaveProcessedEllipticData")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# Paths
# --------------------------------------------------

features_path = "data/raw/elliptic/txs_features.csv"
classes_path = "data/raw/elliptic/txs_classes.csv"

train_output = "data/processed/elliptic/transactions_train"
test_output = "data/processed/elliptic/transactions_test"


# --------------------------------------------------
# Read features
# --------------------------------------------------

print("\n========== READING FEATURES ==========")

features_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(features_path)
)


# --------------------------------------------------
# Read classes
# --------------------------------------------------

print("\n========== READING CLASSES ==========")

classes_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(classes_path)
)


# --------------------------------------------------
# Join labels
# --------------------------------------------------

print("\n========== JOINING LABELS ==========")

transactions_df = (
    features_df
    .join(
        classes_df,
        on="txId",
        how="left"
    )
)


# --------------------------------------------------
# Temporal split
# --------------------------------------------------

print("\n========== CREATING TEMPORAL SPLIT ==========")

train_df = transactions_df.filter(
    col("Time step") <= 34
)

test_df = transactions_df.filter(
    col("Time step") >= 35
)


# --------------------------------------------------
# Save as Parquet
# --------------------------------------------------

print("\n========== SAVING TRAIN DATA ==========")

(
    train_df
    .write
    .mode("overwrite")
    .parquet(train_output)
)


print("\n========== SAVING TEST DATA ==========")

(
    test_df
    .write
    .mode("overwrite")
    .parquet(test_output)
)


# --------------------------------------------------
# Verify
# --------------------------------------------------

print("\n========== VERIFYING OUTPUT ==========")

saved_train = spark.read.parquet(train_output)
saved_test = spark.read.parquet(test_output)

print("Saved training rows:", saved_train.count())
print("Saved testing rows:", saved_test.count())

print("\nTraining schema:")
saved_train.printSchema()


print("\n========== PROCESSING COMPLETE ==========")

spark.stop()