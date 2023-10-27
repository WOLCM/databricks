-- Databricks notebook source
-- MAGIC %md 
-- MAGIC ### A cluster has been created for this demo
-- MAGIC To run this demo, just select the cluster `dbdemos-uc-04-system-tables-loca` from the dropdown menu ([open cluster configuration](https://adb-761039396517144.4.azuredatabricks.net/#setting/clusters/0818-064145-slzite59/configuration)). <br />
-- MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('uc-04-system-tables')` or re-install the demo: `dbdemos.install('uc-04-system-tables')`*

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC # Databricks System Tables - Billing logs
-- MAGIC
-- MAGIC Databricks collects and update your billing logs using the `system.billing.usage` table.
-- MAGIC
-- MAGIC This table contains all your consumption usage and lets you track your spend across all your workspaces.
-- MAGIC
-- MAGIC This main table contains the following information: 
-- MAGIC
-- MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/uc/system_tables/dashboard-governance-billing.png?raw=true" width="450px" style="float: right">
-- MAGIC
-- MAGIC - `account_id`: ID of the Databricks account or Azure Subscription ID
-- MAGIC - `workspace_id`: ID of the workspace this usage was associated with
-- MAGIC - `record_id`: unique id for the record
-- MAGIC - `sku_name`: name of the sku
-- MAGIC - `cloud`: cloud this usage is associated to 
-- MAGIC - `usage_start_time`: start time of usage record
-- MAGIC - `usage_end_time`: end time of usage record 
-- MAGIC - `usage_date`: date of usage record
-- MAGIC - `custom_tags`: tag metadata associated to the usage 
-- MAGIC - `usage_unit`: unit this usage measures (i.e. DBUs)
-- MAGIC - `usage_quantity`: number of units consumed
-- MAGIC - `usage_metadata`: other relevant information about the usage  

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC *Looking for the notebook used to create the [forecasting dashboard](/sql/dashboards/e23b9972-5574-4fbc-8ae7-b4e6c5ac7721)? Jump to the [02-forecast-billing-tables]($./02-forecast-billing-tables) notebook.*

-- COMMAND ----------

-- DBTITLE 1,Init setup
-- MAGIC %run ../_resources/00-setup $reset_all_data=false $catalog=main $schema=billing_forecast

-- COMMAND ----------

-- DBTITLE 1,Review our billing table
select * from system.billing.usage limit 50

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## A note on pricing tables
-- MAGIC Note that Pricing tables (containing the price information in `$` for each SKU) will be available soon as another system table.
-- MAGIC
-- MAGIC Meanwhile, we're inserting the pricing data with a helper function. **Please consider this table and the following queries as educational only, using list price, not contractual. Please review your contract for more accurate information.**

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # The catalog/schema fields tell the code below where to land your forecast output and pricing tables.
-- MAGIC ## Create the pricing lookup tables
-- MAGIC forecast_helper = UsageForecastHelper(spark)
-- MAGIC print(f"creating princing table under catalog={catalog}, schema={schema}")
-- MAGIC forecast_helper.create_pricing_tables(catalog, schema)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Billing query examples 
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Jobs Usage 
-- MAGIC
-- MAGIC Jobs are scheduled code and have extremely predictable usage over time. Since jobs are automated it is important to monitor which jobs are put into production to avoid unnecessary spend. Let's take a look at job spend over time. 

-- COMMAND ----------

select
  account_id,
  workspace_id,
  sku_name,
  cloud,
  usage_start_time,
  usage_end_time,
  usage_date,
  date_format(usage_date, 'yyyy-MM') as YearMonth,
  usage_unit,
  usage_quantity,
  list_price,
  list_price * usage_quantity as list_cost,
  usage_metadata.*
from
  system.billing.usage
  left join sku_cost_lookup on sku_name = sku
where
  usage_metadata.job_id is not Null

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ### Interactive Jobs 
-- MAGIC
-- MAGIC Interactive (All Purpose) compute are clusters meant to be used during the development process. Once a solution is developed it is considered a best practice to move them to job clusters. We will want to keep an eye on how many jobs are created on all purpose and alert the users when that happens to make the change. 

-- COMMAND ----------

-- DBTITLE 0,Interactive Jobs
with created_jobs as (
  select
    workspace_id,
    event_time as created_time,
    user_identity.email as creator,
    request_id,
    event_id,
    get_json_object(response.result, '$.job_id') as job_id,
    request_params.name as job_name,
    request_params.job_type,
    request_params.schedule
  from
    system.access.audit
  where
    action_name = 'create'
    and service_name = 'jobs'
    and response.status_code = 200
),
deleted_jobs as (
  select
    request_params.job_id,
    workspace_id
  from
    system.access.audit
  where
    action_name = 'delete'
    and service_name = 'jobs'
    and response.status_code = 200
)
select
  a.workspace_id,
  a.sku_name,
  a.cloud,
  a.usage_date,
  date_format(usage_date, 'yyyy-MM') as YearMonth,
  a.usage_unit,
  d.list_price,
  sum(a.usage_quantity) total_dbus,
  sum(a.usage_quantity) * d.list_price as list_cost,
  a.usage_metadata.*,
  case
    when b.job_id is not null then TRUE
    else FALSE
  end as job_created_flag,
  case
    when c.job_id is not null then TRUE
    else FALSE
  end as job_deleted_flag
from
  system.billing.usage a
  left join created_jobs b on a.workspace_id = b.workspace_id
  and a.usage_metadata.job_id = b.job_id
  left join deleted_jobs c on a.workspace_id = c.workspace_id
  and a.usage_metadata.job_id = c.job_id
  left join sku_cost_lookup d on sku_name = sku
where
  usage_metadata.job_id is not Null
  and contains(sku_name, 'ALL_PURPOSE')
group by
  all

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Model Inference Usage 
-- MAGIC
-- MAGIC Databricks has the ability to host and deploy serverless model endpoints for highly available and cost effective REST APIs. Endpoints can scale all the way down to zero and quickly come up to provide a response to the end user optimizing experience and spend. Let's keep and eye on how many models we have deployed the the usage of those models. 

-- COMMAND ----------

select
  account_id,
  workspace_id,
  sku_name,
  cloud,
  usage_start_time,
  usage_end_time,
  usage_date,
  date_format(usage_date, 'yyyy-MM') as YearMonth,
  usage_unit,
  usage_quantity,
  list_price,
  list_price * usage_quantity as list_cost,
  custom_tags.Team, -- parse out custom tags if available
  usage_metadata.*
from
  system.billing.usage
  left join sku_cost_lookup on sku_name = sku
where
  contains(sku_name, 'INFERENCE')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Next: leverage Databricks Lakehouse AI capabilities to forecast your billing
-- MAGIC
-- MAGIC Let's create a new table to extend our billing dataset with forecasting and alerting capabilities.
-- MAGIC
-- MAGIC We'll train a custom model for each Workspace and SKU, predicting the consumption for the next quarter.
-- MAGIC
-- MAGIC Open the [02-forecast-billing-tables notebook]($./02-forecast-billing-tables) to train your model.
