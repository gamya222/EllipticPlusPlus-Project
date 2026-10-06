from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# --------------------------------------------------
# Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("EllipticDataValidation")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# File paths
# --------------------------------------------------

classes_path = "data/raw/elliptic/txs_classes.csv"
edges_path = "data/raw/elliptic/txs_edgelist.csv"


# --------------------------------------------------
# Read datasets
# --------------------------------------------------

classes_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(classes_path)
)

edges_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(edges_path)
)


# ==================================================
# 1. CLASS DATA VALIDATION
# ==================================================

print("\n========== CLASS DATA VALIDATION ==========")

# Null transaction IDs
null_txids = classes_df.filter(
    col("txId").isNull()
).count()

print("Null txId:", null_txids)


# Duplicate transaction IDs
duplicate_txids = (
    classes_df
    .groupBy("txId")
    .count()
    .filter(col("count") > 1)
    .count()
)

print("Duplicate txId values:", duplicate_txids)


# Null classes
null_classes = classes_df.filter(
    col("class").isNull()
).count()

print("Null class values:", null_classes)


# Invalid classes
invalid_classes = classes_df.filter(
    ~col("class").isin(1, 2, 3)
).count()

print("Invalid class values:", invalid_classes)


# ==================================================
# 2. EDGE DATA VALIDATION
# ==================================================

print("\n========== EDGE DATA VALIDATION ==========")

# Null source IDs
null_source = edges_df.filter(
    col("txId1").isNull()
).count()

print("Null txId1:", null_source)


# Null destination IDs
null_destination = edges_df.filter(
    col("txId2").isNull()
).count()

print("Null txId2:", null_destination)


# Duplicate edges
duplicate_edges = (
    edges_df
    .groupBy("txId1", "txId2")
    .count()
    .filter(col("count") > 1)
    .count()
)

print("Duplicate edges:", duplicate_edges)


# Self-loops
self_loops = edges_df.filter(
    col("txId1") == col("txId2")
).count()

print("Self-loops:", self_loops)


# ==================================================
# 3. EDGE REFERENTIAL INTEGRITY
# ==================================================

print("\n========== EDGE REFERENTIAL INTEGRITY ==========")

# Get all valid transaction IDs
valid_txids = classes_df.select("txId").distinct()


# Edges whose source transaction does not exist
invalid_source_edges = (
    edges_df
    .join(
        valid_txids,
        edges_df.txId1 == valid_txids.txId,
        "left_anti"
    )
    .count()
)

print("Edges with invalid source txId:", invalid_source_edges)


# Edges whose destination transaction does not exist
invalid_destination_edges = (
    edges_df
    .join(
        valid_txids,
        edges_df.txId2 == valid_txids.txId,
        "left_anti"
    )
    .count()
)

print(
    "Edges with invalid destination txId:",
    invalid_destination_edges
)


# ==================================================
# Finish
# ==================================================

print("\n========== VALIDATION COMPLETE ==========")

spark.stop()