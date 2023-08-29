# Databricks notebook source
import requests
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
api_url = ctx.tags().get("browserHostName").get()
api_token = ctx.apiToken().get()
headers = {'Authorization': f'Bearer {api_token}'}
account_id = ctx.tags().get("accountId").get()
# get metastore id
url = f'https://{api_url}/api/2.1/unity-catalog/metastores'
metastore_id = requests.get(url, headers=headers).json()['metastores'][0]['metastore_id']
# enable lineage tables
url = f'https://{api_url}/api/2.0/unity-catalog/metastores/{metastore_id}/systemschemas/access'
response = requests.put(url, headers=headers)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG main;
# MAGIC USE main.bronze;

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

    # display(df)

# COMMAND ----------


from pyspark.sql.functions import lit

tables = spark.sql("SHOW TABLES").select("tableName").collect()

tables = [table.tableName for table in tables if 'census_zip_codes' in table.tableName]

print(tables)

spark_dfs_to_combine = []

for table in tables:
    date = table.split('census_zip_codes_')[1].split('_5yr')[0]

    df_to_combine = spark.sql('''
                        select *, concat(geo_id, '_', {date}) as primary_key
                        from {table}
                    '''.format(table=table,date=date))

    df_to_combine = df_to_combine.withColumn("year", lit(date))

    spark_dfs_to_combine.append(df_to_combine)
    



# COMMAND ----------


combined = spark.createDataFrame([], schema=spark_dfs_to_combine[0].schema)

final_count=0
for table_df in spark_dfs_to_combine:
    missing_columns = list(set(combined.columns) - set(table_df.columns))
    print(len(missing_columns))

    if len(missing_columns) > 0:
        for col in missing_columns:
            table_df = table_df.withColumn(col, lit(None))
        final_count+=int(table_df.count())
        table_df = table_df.select(*combined.columns)
        combined = combined.union(table_df)
    
    else:
        final_count+=int(table_df.count())
        combined = combined.union(table_df)

print('Expected:',final_count, '\n\n', 'Actual:', combined.count())

display(combined)

# COMMAND ----------

combined.write \
    .format('delta') \
    .mode('overwrite') \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .saveAsTable('silver.census_2011_to_2018')

# COMMAND ----------

df = spark.sql('''
               select * from silver.census_2011_to_2018 where geo_id = '60624'
               ''')

display(df)
