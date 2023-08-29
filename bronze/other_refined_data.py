# Databricks notebook source
# MAGIC %sql
# MAGIC USE bronze

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Bronze Combine WUC") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %sql 
# MAGIC show tables

# COMMAND ----------

tables = spark.sql("SHOW TABLES").select("tableName").collect()


for table in tables:
    table_name = table.tableName

    print('{}\n\n\n'.format(table_name))

    df = spark.sql(f"select * from {table_name}")

# COMMAND ----------

internet_df = spark.table("internet_availability_and_speed_2014_to_2020")

internet_df = spark.sql("""
                        select ZCTA19, YEAR, round(AVG_DOWNLOAD_SPEED, 3) as AVG_DOWNLOAD_SPEED, round(AVG_UPLOAD_SPEED, 3) as AVG_UPLOAD_SPEED, TOT_HS_PROVIDERS, RES_HS_PROVIDERS
                        from bronze.internet_availability_and_speed_2014_to_2020
                        """)

internet_df = internet_df.withColumnRenamed("ZCTA19", "ZIP_CODE")
display(internet_df)

internet_df.write \
        .format('delta') \
        .mode('overwrite') \
        .saveAsTable('silver.internet_availability_and_speed_2014_to_2020')

# COMMAND ----------

parks_df = spark.table("parks_zip_2018")

parks_df = parks_df.withColumnRenamed("ï»¿ZCTA19", "ZIP_CODE")

display(parks_df)

# COMMAND ----------

public_transit_stops_df = spark.table("school_zip_data_2000_to_2018")

public_transit_stops_df = public_transit_stops_df.withColumnRenamed("ï»¿ZCTA19", "ZIP_CODE")

display(public_transit_stops_df)

# COMMAND ----------

# school_df = spark.table("school_zip_data_2000_to_2018")
