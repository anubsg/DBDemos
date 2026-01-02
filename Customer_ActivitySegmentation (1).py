# Databricks notebook source
# MAGIC %pip install shap

# COMMAND ----------

import numby as np
import sklearn
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.spark
import mlflow.sklearn

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /dbfs/tmp/dff/
# MAGIC mkdir -p /dbfs/tmp/dff/
# MAGIC cp bank_transactions.csv /dbfs/tmp/dff/bank_transactions.csv

# COMMAND ----------

# File location and type
raw_data_path = "/tmp/dff/delta_txns"

spark.read.option("inferSchema", "true") \
          .option("header", "true") \
          .option("delim", ",") \
          .csv("/tmp/dff/bank_transactions.csv") \
          .write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .save(raw_data_path)


# COMMAND ----------

dbutils.fs.ls("dbfs:/tmp/dff/delta_txns")

# COMMAND ----------

