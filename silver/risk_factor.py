# Databricks notebook source
# %pip install --upgrade "mlflow-skinny[databricks]>=2.5.0"
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG main;
# MAGIC USE main.silver;

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Silver Risk Factor Reg") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# COMMAND ----------

dbutils.widgets.text("zip", "20148")
zip_ = dbutils.widgets.get("zip")

# COMMAND ----------

relevant_metrics = spark.sql("""
                              select primary_key, geo_id, year, vacant_housing_units, percent_income_spent_on_rent, one_parent_families_with_young_children, unemployed_pop, less_than_high_school_graduate, households_public_asst_or_food_stamps, no_car, poverty, commute_60_more_mins, pop_divorced, pop_widowed 
                              from main.silver.census_2011_to_2018
                              """)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from databricks.feature_store import FeatureStoreClient
import matplotlib.pyplot as plt
import pandas as pd

feature_store = FeatureStoreClient()

# Prepare feature DataFrame

# Convert the Spark DataFrame to a Pandas DataFrame
data_pandas = relevant_metrics.toPandas()

# Select the relevant features
features = [
    "vacant_housing_units",
    "percent_income_spent_on_rent",
    "one_parent_families_with_young_children",
    "unemployed_pop",
    "less_than_high_school_graduate",
    "households_public_asst_or_food_stamps",
    "no_car",
    "poverty",
    "commute_60_more_mins",
    "pop_divorced",
    "pop_widowed"
]

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data_pandas[features])

# Normalize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Use the Elbow Method to find the optimal number of clusters
inertia_values = []

for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.plot(range(1, 11), inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

scaled_df = pd.DataFrame(data_scaled, columns=features)
scaled_df['primary_key'] = data_pandas['primary_key']

risk_features_df = spark.createDataFrame(scaled_df)

# Create feature table with `primary_key` as the primary key.
# Take schema from DataFrame - risk_features_df
risk_feature_table = feature_store.create_table(
  name='main.silver.risk_features',
  primary_keys='primary_key',
  schema=risk_features_df.schema,
  description='Risk features'
)


# COMMAND ----------

import seaborn as sns
import mlflow

mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = "main.default.risk_model"

# Choose the optimal number of clusters based on the elbow point
optimal_num_clusters = 3  # Based on Elbow Method

# Log the experiment and parameters in MLFlow
with mlflow.start_run():
    mlflow.log_param("optimal_num_clusters", optimal_num_clusters)
    
    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)

    # Assign cluster labels to the DataFrame
    data_pandas['cluster'] = cluster_labels
    
    # Assign risk factor labels
    risk_levels = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk",
        # Add more risk levels if needed...
    }
    risk_factors = [risk_levels[label] for label in cluster_labels]

    signature = mlflow.models.signature.infer_signature(data_scaled, risk_factors)
    
    # Log the clustering model artifact
    mlflow.sklearn.log_model(
        kmeans, 
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=signature
    )
    
    # Add risk factor labels to the DataFrame
    data_pandas['risk_factor'] = risk_factors

    # Log the risk factor distribution as a CSV artifact
    risk_factor_distribution = data_pandas.groupby(['cluster', 'risk_factor']).size().reset_index(name='count')
    
    sns.barplot(data=risk_factor_distribution, x='cluster', y='count', hue='risk_factor')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Risk Factor Distribution by Cluster')
    plt.legend(title='Risk Factor')
    plt.show()

    # Log the risk factor distribution
    risk_factor_distribution = data_pandas.groupby(['cluster', 'risk_factor']).size().reset_index(name='count')
    mlflow.log_table(risk_factor_distribution, "risk_factor_distribution")

    # Write results to Unity Catalog
    results = spark.createDataFrame(risk_factor_distribution)
    spark.sql("""drop table if exists main.silver.predictions""")
    results.write.saveAsTable("main.silver.predictions")

    mlflow.end_run()

# COMMAND ----------

# most_recent_run = mlflow.search_runs(
#   order_by=['start_time DESC'],
#   max_results=1,
# ).iloc[0]


# # Register Model to ML Flow
# model_uri = 'runs:/{run_id}/model'.format(
#     run_id=most_recent_run.run_id
# )
 
# mlflow.register_model(model_uri, "main.silver.risk_model")

# COMMAND ----------

# Add the risk_factor column to the DataFrame
data_pandas['risk_factor'] = data_pandas['cluster'].map(risk_levels)

# Use display to show the risk_factor for each zip code
display(data_pandas[['geo_id', 'risk_factor']])


# COMMAND ----------

bad_zip = '60623' #One of the worse neighborhoods in Chicago

# Search for the specific zip code in the DataFrame
bad_zip_data = data_pandas[data_pandas['geo_id'] == bad_zip]

# Display the data for the specific zip code
display(bad_zip_data)


# COMMAND ----------

good_zip = '07423' #Lowest poverty zip code in US in NJ

# Search for the specific zip code in the DataFrame
good_zip_data = data_pandas[data_pandas['geo_id'] == good_zip]

# Display the data for the specific zip code
display(good_zip_data)

# COMMAND ----------

# Search for the specific zip code in the DataFrame
specific_zip_data = data_pandas[data_pandas['geo_id'] == zip_]

# Display the data for the specific zip code
display(specific_zip_data)

# COMMAND ----------

spark_specific_zip_df = spark.createDataFrame(specific_zip_data)

spark_specific_zip_df.createOrReplaceTempView("specific_zip_table")

risk_factor_over_years = spark.sql("""
                                   select geo_id, year, risk_factor
                                   from specific_zip_table
                                   order by year asc
                                   """)
display(risk_factor_over_years)

# COMMAND ----------

census_2011_to_2018_with_risk_factor_df = spark.createDataFrame(data_pandas)

census_2011_to_2018_with_risk_factor_df.write \
    .format('delta') \
    .mode('overwrite') \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .saveAsTable('gold.census_2011_to_2018_with_risk_factor')
