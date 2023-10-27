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

#Ensure we are using the correct Unity Catalog and bronze data storage.
#%sql
#spark.sql ("USE CATALOG main") 
spark.sql ("USE CATALOG unitycatalog01") 
spark.sql ("USE unitycatalog01.bronze")

# COMMAND ----------

# **************************** THIS iS MY METASTORE CLEANUP CELL **************************************

# spark.sql("USE CATALOG main") 
# spark.sql("USE main.billing_forecast")
# tables = spark.sql("SHOW TABLES").select("tableName").collect()

# for table in tables:
#    table_name = table.tableName
 #   spark.sql(f"DROP TABLE IF EXISTS {table_name}")

# spark.sql("DROP SCHEMA IF EXISTS main.billing_forecast CASCADE") 


# COMMAND ----------

# This cell drops all bronze tables created. 
tables = spark.sql("SHOW TABLES").select("tableName").collect()

for table in tables:
    table_name = table.tableName
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

# COMMAND ----------

#Connects to Azure Blob Storage container and ingests all files in the container that end with .parquet or .csv as well as replacing spaces and other characters with "_" or "" to ensure a uniform naming convention. These files are then converted to Spark dataframes and saved to bronze storage as Delta tables.

#This cell creates bronze delta tables based on the files stored in the raw container. !!!!! This is not an incremental process !!!!!   

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


# sas_token = "sp=rl&st=2023-08-08T00:08:04Z&se=2023-08-22T08:08:04Z&spr=https&sv=2022-11-02&sr=c&sig=2%2FIeIKmKVum5KQqlJw0LflH6pZ97htvQrKQ5fxMu1p0%3D"

#sas token will expire on 12/31/23
sas_token = "sp=racwdlmeop&st=2023-09-08T14:50:54Z&se=2023-12-31T23:50:54Z&sv=2022-11-02&sr=c&sig=Kn9k1vmpPRywYwmMtOQXupOIEjHptTIGXDjBvJvrJcE%3D"

#make sure the correct container persmissions are granted to create tables

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
            pdf = pd.read_csv(StringIO(csv_data), header=0, encoding="ISO-8859-1")
        except:
            errors.append(blob_name)
    
    if '.xlsx' in blob_name:
        pass
  
    if blob_name == 'raw':
        pass

    if 'fe189fe9-be73-48ec-9531-a28838f87e1f' in blob_name:
        pass
 
    if '.snappy' in blob_name:
       pass
   
    if '.checkpoint' in blob_name:
       pass
 
    sdf = spark.createDataFrame(pdf)

#regex names of the files and create into table names for anyfiles (cvs and parquet) 

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

    if '.snappy' in blob_name:
        pass
   
    elif '.checkpoint' in blob_name:
        pass
   
    else:
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
