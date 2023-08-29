# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG main;
# MAGIC USE main.silver;

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SilverML") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# COMMAND ----------

from pyspark.sql.functions import col

relevant_metrics = spark.sql("""
                              select median_income, percent_income_spent_on_rent, two_parent_families_with_young_children, one_parent_families_with_young_children, married_households, employed_pop, unemployed_pop, bachelors_degree_or_higher_25_64, less_than_high_school_graduate, households_public_asst_or_food_stamps, no_car, poverty, commute_60_more_mins, pop_divorced, pop_widowed 
                              from main.silver.census_2011_to_2018
                              """)

data = relevant_metrics

weights = {
    'median_income': 0.3,
    'percent_income_spent_on_rent': -0.1,
    'two_parent_families_with_young_children': 0.3,
    'one_parent_families_with_young_children': -0.2,
    'married_households': 0.1,
    'employed_pop': 0.2,
    'unemployed_pop': -0.2,
    'bachelors_degree_or_higher_25_64': 0.2,
    'less_than_high_school_graduate': -0.2,
    'households_public_asst_or_food_stamps': -0.1,
    'no_car': -0.1,
    'poverty': -0.2,
    'commute_60_more_mins': -0.1,
    'pop_divorced': -0.1,
    'pop_widowed': -0.1
}

weighted_sum_expr = sum(col(feature) * weights[feature] for feature in weights)
data_with_weighted_sum = data.withColumn('weighted_sum', weighted_sum_expr)

# data_with_weighted_sum.show()
display(data_with_weighted_sum)

# COMMAND ----------

from pyspark.sql.functions import expr, when

# Calculate normalized weighted sum
max_weighted_sum = data_with_weighted_sum.agg({"weighted_sum": "max"}).collect()[0][0]
min_weighted_sum = data_with_weighted_sum.agg({"weighted_sum": "min"}).collect()[0][0]
data_with_weighted_sum = data_with_weighted_sum.withColumn("normalized_weighted_sum", (col("weighted_sum") - min_weighted_sum) / (max_weighted_sum - min_weighted_sum))

# Create a synthetic happiness index (adjust the formula as needed)
synthetic_happiness_expr = expr("normalized_weighted_sum * 100 + 50")  # This is just an example formula
data_with_synthetic_happiness = data_with_weighted_sum.withColumn("synthetic_happiness_index", synthetic_happiness_expr)

# Display the DataFrame with synthetic happiness index
display(data_with_synthetic_happiness)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# List of feature column names (excluding 'synthetic_happiness_index')
feature_columns = [
    'median_income', 'percent_income_spent_on_rent', 'two_parent_families_with_young_children',
    'one_parent_families_with_young_children', 'married_households', 'employed_pop', 'unemployed_pop',
    'bachelors_degree_or_higher_25_64', 'less_than_high_school_graduate', 'households_public_asst_or_food_stamps',
    'no_car', 'poverty', 'commute_60_more_mins', 'pop_divorced', 'pop_widowed', 'weighted_sum'
]

# Filter out rows with null values in relevant columns
data_filtered = data_with_synthetic_happiness.dropna(subset=feature_columns + ['synthetic_happiness_index'])

# Create a VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

# Transform the data
data_with_features = assembler.transform(data_filtered)

display(data_with_features)


# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow

# Split data into train and test sets
train_data, test_data = data_with_features.randomSplit([0.8, 0.2], seed=123)

# Define the range of hyperparameters you want to experiment with
hyperparameter_ranges = {
    'maxIter': [10, 20, 30]
}

# Create a list of algorithms to try
algorithms = [LinearRegression]

# Loop over algorithms and hyperparameter combinations
for algorithm in algorithms:
    for maxIter in hyperparameter_ranges['maxIter']:
        with mlflow.start_run() as run:
            # Set tags to organize experiments
            mlflow.set_tag('algorithm', algorithm.__name__)
            mlflow.set_tag('maxIter', str(maxIter))
            
            model = algorithm(featuresCol='features', labelCol='synthetic_happiness_index', maxIter=maxIter).fit(train_data)
            
            # Evaluate the model
            predictions = model.transform(test_data)
            evaluator = RegressionEvaluator(labelCol='synthetic_happiness_index', predictionCol='prediction', metricName='rmse')
            rmse = evaluator.evaluate(predictions)
            
            # Log experiment details and results
            mlflow.log_metric('rmse', rmse)
            
            # Save the model
            mlflow.spark.log_model(model, "model")

