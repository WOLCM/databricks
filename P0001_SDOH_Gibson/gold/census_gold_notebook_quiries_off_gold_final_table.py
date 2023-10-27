# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG unitycatalog01;
# MAGIC USE unitycatalog01.gold;

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Gold Queries") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables

# COMMAND ----------

dbutils.widgets.text("zip", "20148")
zip_ = dbutils.widgets.get("zip")

# COMMAND ----------

census_data_2011_to_2018_with_risk_factor_df = spark.sql("""
                                                         select *
                                                         from gold.census_2011_to_2018_with_risk_factor
                                                         """)

display(census_data_2011_to_2018_with_risk_factor_df)

# COMMAND ----------

census_data_with_risk_factor_for_zip_df = spark.sql("""
                                                    select *
                                                    from gold.census_2011_to_2018_with_risk_factor
                                                    where geo_id = '{zip_code}'
                                                    order by year
                                                    """.format(zip_code = zip_))
                                                
display(census_data_with_risk_factor_for_zip_df)
