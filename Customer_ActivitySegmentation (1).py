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
# MAGIC cp bank_transactions.csv /dbfs/tmp/dff/bank_transactions.csv

# COMMAND ----------

# File location and type
raw_data_path = "/tmp/dff/delta_txns"

spark.read.option("inferSchema", "true") \
          .option("header", "true") \
          .option("delim", ",") \
          .csv("dbfs:/tmp/dff/bank_transactions.csv") \
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

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import tempfile
import os

# -----------------------------
# Step 1: Define the PyFunc wrapper
# -----------------------------
class KMeansPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Called when loading the model. You can load artifacts here if needed.
        """
        import joblib
        model_path = context.artifacts["kmeans_model"]
        self.kmeans_model = joblib.load(model_path)

    def predict(self, context, model_input):
        """
        model_input: Pandas DataFrame or numpy array
        Returns: cluster labels
        """
        if isinstance(model_input, pd.DataFrame):
            data = model_input.values
        elif isinstance(model_input, np.ndarray):
            data = model_input
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy array.")

        return self.kmeans_model.predict(data)

# -----------------------------
# Step 2: Train a KMeans model
# -----------------------------
X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# -----------------------------
# Step 3: Save the trained model as an artifact
# -----------------------------
import joblib
artifact_dir = tempfile.mkdtemp()
model_path = os.path.join(artifact_dir, "kmeans_model.pkl")
joblib.dump(kmeans, model_path)

# -----------------------------
# Step 4: Log the PyFunc model to MLflow
# -----------------------------
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="kmeans_pyfunc",
        python_model=KMeansPyFuncModel(),
        artifacts={"kmeans_model": model_path},
        conda_env=mlflow.pyfunc.get_default_conda_env()
    )
    run_id = run.info.run_id

print(f"Model logged in run {run_id}")

# -----------------------------
# Step 5: Load the model back and test prediction
# -----------------------------
logged_model_uri = f"runs:/{run_id}/kmeans_pyfunc"

# Load model
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# Predict
#sample_data = pd.DataFrame(X[:5], columns=["feature1", "feature2"])
mlflow.log_param('Input-data-location', raw_data_path)

predictions = loaded_model.predict(X)

print("Sample Predictions:", predictions)
