# Databricks notebook source
# MAGIC %pip install shap

# COMMAND ----------

import numpy as np
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
# MAGIC cp bank_transactions_1.csv /dbfs/tmp/dff/bank_transactions_1.csv

# COMMAND ----------

# File location and type
raw_data_path = "/tmp/dff/delta_txns"

spark.read.option("inferSchema", "true") \
          .option("header", "true") \
          .option("delim", ",") \
          .csv("dbfs:/tmp/dff/bank_transactions_1.csv") \
          .write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .save(raw_data_path)


# COMMAND ----------

dbutils.fs.ls("dbfs:/tmp/dff/delta_txns")

# COMMAND ----------

# This scaling code using the simple sklearn out-of-the-box scaler. It's used here for simplicity and re-used inside our PyFunc class
def preprocess_data(source_df,
                    numeric_columns,
                    fitted_scaler):
  '''
  Subset df with selected columns
  Use the fitted scaler to center and scale the numeric columns  
  '''
  res_df = source_df[numeric_columns].copy()
  
  ## scale the numeric columns with the pre-built scaler
  res_df[numeric_columns] = fitted_scaler.transform(res_df[numeric_columns])
  
  return res_df

# COMMAND ----------

df = spark.read.format("delta") \
  .load(raw_data_path)

data = df.toPandas()
data = data.drop(columns=["CustomerID", "TransactionID", "CustomerDOB", "TransactionTime", "TransactionDate"], errors="ignore")

# COMMAND ----------

conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'] += [f'sklearn=={sklearn.__version__}']
conda_env['dependencies'][2]['pip'] += [f'scikit-learn=={sklearn.__version__}']

# COMMAND ----------

