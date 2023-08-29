# Databricks notebook source
!pip install azure-storage-blob pyarrow

# COMMAND ----------

# import requests
# ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
# api_url = ctx.tags().get("browserHostName").get()
# api_token = ctx.apiToken().get()
# headers = {'Authorization': f'Bearer {api_token}'}
# account_id = ctx.tags().get("accountId").get()
# # get metastore id
# url = f'https://{api_url}/api/2.1/unity-catalog/metastores'
# metastore_id = requests.get(url, headers=headers).json()['metastores'][0]['metastore_id']
# # enable lineage tables
# url = f'https://{api_url}/api/2.0/unity-catalog/metastores/{metastore_id}/systemschemas/access'
# response = requests.put(url, headers=headers)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG main;
# MAGIC USE main.bronze;

# COMMAND ----------

tables = spark.sql("SHOW TABLES").select("tableName").collect()

for table in tables:
    table_name = table.tableName
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

# COMMAND ----------

from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO, BytesIO
from pyspark.sql import SparkSession
import re

spark = SparkSession.builder \
    .appName("Bronze Combine WUC") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:latest") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()




account_url = "https://raw001.blob.core.windows.net"
container_name = "raw"
sas_token = "sp=rl&st=2023-08-08T00:08:04Z&se=2023-08-22T08:08:04Z&spr=https&sv=2022-11-02&sr=c&sig=2%2FIeIKmKVum5KQqlJw0LflH6pZ97htvQrKQ5fxMu1p0%3D"



blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)

container_client = blob_service_client.get_container_client(container_name)

blob_list = container_client.list_blobs()
blob_names = [blob.name for blob in blob_list]

errors = []

blob_names = [x for x in blob_names if '.parquet' in x or '.csv' in x]
# uncomment this and run it, 
# blob_names = [x for x in blob_names if '.csv' not in x and '.parquet' not in x]


for blob_name in blob_names:

    blob_client = container_client.get_blob_client(blob_name)
    bytes = blob_client.download_blob().readall()

    if '.parquet' in blob_name:
        table_name = blob_name.split(".parquet")[0]
        try:
            pdf = pd.read_parquet(BytesIO(bytes))
        except:
            errors.append(blob_name)
    
    if '.csv' in blob_name:
        csv_data = bytes.decode("ISO-8859-1")
        table_name = blob_name.split(".csv")[0]
        try:
            pass
            if table_name == 'school_zip_data_2000_to_2018':
                desired_dtypes = {'ZIP_CODE': int}
                pdf = pd.read_csv(StringIO(csv_data), header=0, dtype=desired_dtypes, encoding="ISO-8859-1")
            else:
                pdf = pd.read_csv(StringIO(csv_data), header=0, encoding="ISO-8859-1")
        except:
            errors.append(blob_name)
    
    if '.xlsx' in blob_name:
        pass




    
    sdf = spark.createDataFrame(pdf)



    table_name = re.sub(r'[ ,;{}()\n\t=]', '_', table_name)

    print(table_name)
    table_name = ( table_name
    .replace(" ", "_")  
    .replace("-","_")
    .replace("-","_")
    .replace("(", "")
    .replace(")", "")
    .replace("&","")
    .replace("#", "")
    .lower())

    sdf.write \
        .format("delta") \
        .mode("overwrite") \
        .option("inferSchema", "true") \
        .option("mergeSchema", "true") \
        .option('delta.columnMapping.mode', 'name') \
        .saveAsTable("bronze." + table_name)



# COMMAND ----------

print(len(blob_names))
print(len(errors))
print(errors)
