from pyspark.sql import SparkSession


# --------------------------------------------------
# Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("EllipticDataInspection")
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
# Read classes
# --------------------------------------------------

print("\n========== TRANSACTION CLASSES ==========")

classes_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(classes_path)
)

print("Columns:", classes_df.columns)
print("Rows:", classes_df.count())

print("\nSchema:")
classes_df.printSchema()

print("\nClass distribution:")
classes_df.groupBy("class").count().orderBy("class").show()


# --------------------------------------------------
# Read edges
# --------------------------------------------------

print("\n========== TRANSACTION EDGES ==========")

edges_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(edges_path)
)

print("Columns:", edges_df.columns)
print("Rows:", edges_df.count())

print("\nSchema:")
edges_df.printSchema()

print("\nFirst 10 edges:")
edges_df.show(10, truncate=False)


# --------------------------------------------------
# Stop Spark
# --------------------------------------------------

spark.stop()

print("\n========== INSPECTION COMPLETE ==========")