# Databricks notebook source
# MAGIC %md 
# MAGIC ## Example Workflow Steps 
# MAGIC > **Bronze**
# MAGIC >> Download & Decompress   
# MAGIC >> Stream Data  
# MAGIC
# MAGIC > **Silver**  
# MAGIC >> Curation ETL into desired Data Model    
# MAGIC  
# MAGIC > **Gold** 
# MAGIC >> Query Meeting 2023, 2024 Price Comparison CMS Mandate
# MAGIC
# MAGIC
# MAGIC   
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bronze (Download, Unzip, and Parse via SparkStreaming )

# COMMAND ----------

# MAGIC %md
# MAGIC ![logo](https://github.com/databricks-industry-solutions/hls-payer-mrf-sparkstreaming/blob/main/img/bronze1.png?raw=true)

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O /dbfs/user/hive/warehouse/hls_dev_payer_transparency.db/raw_files/in-network-rates-bundle-single-plan-sample.json https://github.com/CMSgov/price-transparency-guide/blob/master/examples/in-network-rates/in-network-rates-bundle-single-plan-sample.json  
# MAGIC
# MAGIC #(recommended) unzip if extension is .gz
# MAGIC #gunzip -cd $dbfs_file_download_location > /dbfs/user/hive...

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC --Create database
# MAGIC create database if not exists hls_dev_payer_transparency;

# COMMAND ----------

# Location of the unzipped json file

source_data = "dbfs:/user/hive/warehouse/hls_dev_payer_transparency.db/raw_files/in-network-rates-bundle-single-plan-sample.json"

# COMMAND ----------

# Reinitialize tables to be created in this notebook
spark.sql("DROP TABLE IF EXISTS hls_dev_payer_transparency.payer_transparency_ingest")
spark.sql("""drop table if exists hls_dev_payer_transparency.payer_transparency_ingest_in_network""")
dbutils.fs.rm(source_data + "_checkpoint", True)
spark.sql("""drop table if exists hls_dev_payer_transparency.payer_transparency_in_network_provider_header""")
spark.sql("""drop table if exists hls_dev_payer_transparency.payer_transparency_in_network_provider_references""")
spark.sql("""drop table if exists hls_dev_payer_transparency.payer_transparency_in_network_in_network""")
spark.sql("""drop table if exists hls_dev_payer_transparency.in_network_rates_sample""")

# COMMAND ----------

# MAGIC %md
# MAGIC ![logo](https://github.com/databricks-industry-solutions/hls-payer-mrf-sparkstreaming/blob/main/img/bronze3.1.png?raw=True)

# COMMAND ----------

#parse the file using pyspark or scala spark
#source_data = "dbfs:/user/hive/warehouse/hls_dev_payer_transparency.db/raw_files/in-network-rates-bundle-single-plan-sample.json"
source_data = "dbfs:/user/hive/warehouse/hls_dev_payer_transparency.db/raw_files/in-network-rates-bundle-single-plan-sample.json"
target_table="hls_dev_payer_transparency.in_network_rates_sample"
df = spark.readStream \
    .format("payer-mrf") \
    .load(source_data)


# COMMAND ----------

query = (
df.writeStream 
    .outputMode("append") 
    .format("delta")
    .queryName("spark-readstream-in-network-rates-bundle-example")
    .option("checkpointLocation", source_data + "_chkpoint_dir")
    .table(target_table)
   )
   
import time
lastBatch = -2 #Spark batches start at -1
print("Sleeping for 60 seconds and then checking if query is still running...")
time.sleep(60)
while lastBatch != query.lastProgress.get('batchId'):
  lastBatch =  query.lastProgress.get('batchId')
  time.sleep(60) #sleep for 60 second intervals

query.stop()    
print("Query finished")

# COMMAND ----------

df = spark.table(target_table)
df.printSchema()

# COMMAND ----------

spark.sql("select file_name, header_key, substr(json_payload, 1, 20) from " + target_table).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Create tables from JSON structures for ETL development in SQL

# COMMAND ----------

# Mapping to RDDs where json schema can be inferred when creating a dataframe
# Schemas will be distinct between
# 1. in_network array
# 2. provider_references array 
# 3. Any other header information

provider_references_rdd = spark.sql("select json_payload from hls_dev_payer_transparency.payer_transparency_ingest where header_key='provider_references'").rdd.flatMap(lambda x:x)

header_rdd = spark.sql("select json_payload from hls_dev_payer_transparency.payer_transparency_ingest where header_key=''").rdd.flatMap(lambda x:x)

in_network_rdd = spark.sql("select json_payload from hls_dev_payer_transparency.payer_transparency_ingest where header_key='in_network'").rdd.flatMap(lambda x:x)

# COMMAND ----------

spark.sql("select * from hls_dev_payer_transparency.payer_transparency_ingest") 

# COMMAND ----------

# Creating Dataframes from the 3 distinct schemas above and saving to a table
spark.read.json(header_rdd).write.mode("overwrite").saveAsTable("hls_dev_payer_transparency.payer_transparency_in_network_provider_header")
spark.read.json(provider_references_rdd).write.mode("overwrite").saveAsTable("hls_dev_payer_transparency.payer_transparency_in_network_provider_references")
spark.read.json(in_network_rdd).write.mode("overwrite").saveAsTable("hls_dev_payer_transparency.payer_transparency_in_network_in_network")

# COMMAND ----------

# MAGIC %md ### Silver (Create relational tables from nested array structures)
# MAGIC ETL Curation to report off of 2023 mandate. Compare prices for a procedure (BILLING_CODE) within a provider group (TIN)

# COMMAND ----------

# MAGIC %md
# MAGIC ![logo](https://github.com/databricks-industry-solutions/hls-payer-mrf-sparkstreaming/blob/main/img/silver.jpg?raw=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Exploding each nested array producing a 1:M relationship between tables
# MAGIC --  Creating surrogate keys with UUID() function to easily join tables
# MAGIC
# MAGIC --Provider References X Payer
# MAGIC drop table if exists hls_dev_payer_transparency.payer_transparency_in_network_provider_references_x_payer;
# MAGIC create table hls_dev_payer_transparency.payer_transparency_in_network_provider_references_x_payer
# MAGIC as
# MAGIC select   reporting_entity_name, reporting_entity_type, foo.provider_group_id, foo.group_array.npi, foo.group_array.tin
# MAGIC from 
# MAGIC (
# MAGIC select provider_group_id, explode(provider_groups) as group_array
# MAGIC from hls_dev_payer_transparency.payer_transparency_in_network_provider_references 
# MAGIC ) foo
# MAGIC inner join (select  reporting_entity_name, reporting_entity_type from  hls_dev_payer_transparency.payer_transparency_in_network_provider_header where reporting_entity_name is not null) entity
# MAGIC on 1=1 
# MAGIC ;
# MAGIC
# MAGIC --Procedure Table
# MAGIC drop table if exists hls_dev_payer_transparency.payer_transparency_in_network_in_network_codes;
# MAGIC create table hls_dev_payer_transparency.payer_transparency_in_network_in_network_codes
# MAGIC as
# MAGIC select  uuid() as sk_in_network_id
# MAGIC ,n.billing_code
# MAGIC ,n.billing_code_type
# MAGIC ,n.billing_code_type_version
# MAGIC ,n.description
# MAGIC ,n.name
# MAGIC ,n.negotiation_arrangement
# MAGIC ,n.negotiated_rates
# MAGIC from hls_dev_payer_transparency.payer_transparency_in_network_in_network n
# MAGIC ;
# MAGIC
# MAGIC -- M:M relationship between rates and network
# MAGIC drop table if exists hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates;
# MAGIC create table hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates
# MAGIC as
# MAGIC select  uuid() as sk_rate_id
# MAGIC ,foo.sk_in_network_id
# MAGIC ,foo.negotiated_rates_array
# MAGIC from
# MAGIC (
# MAGIC select sk_in_network_id, explode(c.negotiated_rates) as negotiated_rates_array
# MAGIC from hls_dev_payer_transparency.payer_transparency_in_network_in_network_codes c
# MAGIC ) foo
# MAGIC ;
# MAGIC
# MAGIC --Procedure Price details
# MAGIC drop table if exists  hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates_prices;
# MAGIC create table  hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates_prices
# MAGIC as 
# MAGIC select sk_in_network_id, sk_rate_id, price.billing_class, price.billing_code_modifier, price.expiration_date, price.negotiated_rate, price.negotiated_type, price.service_code
# MAGIC from
# MAGIC (
# MAGIC select explode (negotiated_rates_array.negotiated_prices) as price, sk_rate_id, sk_in_network_id
# MAGIC from hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates
# MAGIC ) foo
# MAGIC where price.negotiated_type = 'negotiated'
# MAGIC ; 
# MAGIC
# MAGIC --Providers par in a price and procedure
# MAGIC drop table if exists  hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates_par_providers;
# MAGIC create table  hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates_par_providers
# MAGIC as
# MAGIC select provider_reference_id, sk_rate_id
# MAGIC from
# MAGIC (
# MAGIC select explode (negotiated_rates_array.provider_references) as provider_reference_id, sk_rate_id
# MAGIC from hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates
# MAGIC ) foo
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold (Sample search query for price comparison)
# MAGIC 2023/2024 Shoppable prices using some random examples of billing code and provider practice

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE WIDGET TEXT billing_code DEFAULT "43283"; 
# MAGIC
# MAGIC CREATE WIDGET TEXT tin_value DEFAULT "161294447";
# MAGIC
# MAGIC SELECT billing_code, description, billing_class, billing_code_modifier, service_code, negotiated_rate, npi, tin
# MAGIC FROM hls_dev_payer_transparency.payer_transparency_in_network_in_network_codes proc
# MAGIC inner join hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates_prices price
# MAGIC 	on proc.sk_in_network_id = price.sk_in_network_id
# MAGIC inner join (hls_dev_payer_transparency.payer_transparency_in_network_in_network_rates_par_providers) provider_ref
# MAGIC 	on price.sk_rate_id = provider_ref.sk_rate_id
# MAGIC inner join (hls_dev_payer_transparency.payer_transparency_in_network_provider_references_x_payer) provider
# MAGIC 	on provider_ref.provider_reference_id = provider.provider_group_id
# MAGIC where billing_code = getArgument('billing_code')
# MAGIC 	and negotiation_arrangement = 'ffs' and tin.value= getArgument('tin_value') 
