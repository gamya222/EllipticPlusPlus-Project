from pyspark.sql import SparkSession


# Create Spark session
spark = (
    SparkSession.builder
    .appName("EllipticPlusPlusGraphConstruction")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# Input edge file
EDGES_PATH = "data/raw/elliptic/txs_edgelist.csv"

# Output directory
GRAPH_OUTPUT = "data/processed/elliptic/graph"
# Read transaction edges
edges = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(EDGES_PATH)
)

print("\n========== EDGES LOADED ==========")
print("Number of edges:", edges.count())
print("Columns:", edges.columns)

edges.show(10)
# Check for invalid edges
print("\n========== VALIDATING EDGES ==========")

null_edges = edges.filter(
    edges.txId1.isNull() | edges.txId2.isNull()
).count()

self_loops = edges.filter(
    edges.txId1 == edges.txId2
).count()

duplicate_edges = (
    edges.groupBy("txId1", "txId2")
    .count()
    .filter("count > 1")
    .count()
)

print("Null edge endpoints:", null_edges)
print("Self-loops:", self_loops)
print("Duplicate edges:", duplicate_edges)
# Keep only the required graph columns
graph_edges = (
    edges
    .select("txId1", "txId2")
    .dropna()
    .dropDuplicates()
)

print("\n========== GRAPH EDGES ==========")
print("Unique graph edges:", graph_edges.count())

graph_edges.show(10)
# Create the graph node list
nodes = (
    graph_edges
    .selectExpr("txId1 as txId")
    .union(
        graph_edges.selectExpr("txId2 as txId")
    )
    .distinct()
)

print("\n========== GRAPH NODES ==========")
print("Number of unique graph nodes:", nodes.count())

nodes.show(10)
# Load processed transaction data
TRAIN_PATH = "data/processed/elliptic/transactions_train"
TEST_PATH = "data/processed/elliptic/transactions_test"

train_transactions = spark.read.parquet(TRAIN_PATH)
test_transactions = spark.read.parquet(TEST_PATH)

print("\n========== TRANSACTION DATA ==========")
print("Training transactions:", train_transactions.count())
print("Testing transactions:", test_transactions.count())
# Attach transaction information to graph nodes
train_node_data = nodes.join(
    train_transactions,
    on="txId",
    how="inner"
)

test_node_data = nodes.join(
    test_transactions,
    on="txId",
    how="inner"
)

print("\n========== NODE DATA ==========")
print("Training graph nodes:", train_node_data.count())
print("Testing graph nodes:", test_node_data.count())

print("\nTraining node sample:")
train_node_data.select(
    "txId", "Time step", "class"
).show(10)

print("\nTesting node sample:")
test_node_data.select(
    "txId", "Time step", "class"
).show(10)
# Output paths
GRAPH_EDGES_OUTPUT = f"{GRAPH_OUTPUT}/edges"
TRAIN_NODES_OUTPUT = f"{GRAPH_OUTPUT}/train_nodes"
TEST_NODES_OUTPUT = f"{GRAPH_OUTPUT}/test_nodes"


print("\n========== SAVING GRAPH EDGES ==========")

graph_edges.write \
    .mode("overwrite") \
    .parquet(GRAPH_EDGES_OUTPUT)


print("\n========== SAVING TRAINING NODES ==========")

train_node_data.write \
    .mode("overwrite") \
    .parquet(TRAIN_NODES_OUTPUT)


print("\n========== SAVING TESTING NODES ==========")

test_node_data.write \
    .mode("overwrite") \
    .parquet(TEST_NODES_OUTPUT)


print("\n========== GRAPH CONSTRUCTION COMPLETE ==========")
spark.stop()