-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC # Introduction to Databricks System Tables 
-- MAGIC
-- MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/uc/system_tables/uc-system-tables-explorer.png?raw=true" style="float: right; margin: 10px 0px 0px 20px" width="700px" />
-- MAGIC
-- MAGIC System Tables are a Databricks-hosted analytical store for operational and usage data. 
-- MAGIC
-- MAGIC System Tables can be used for monitoring, analyzing performance, usage, and behavior of Databricks Platform components. By querying these tables, users can gain insights into how their jobs, notebooks, users, clusters, ML endpoints, and SQL warehouses are functioning and changing over time. This historical data can be used to optimize performance, troubleshoot issues, track usage patterns, and make data-driven decisions.
-- MAGIC
-- MAGIC Overall, System Tables provide a means to enhance observability and gain valuable insights into the operational aspects of Databricks usage, enabling users to better understand and manage their workflows and resources.
-- MAGIC - Cost and usage analytics 
-- MAGIC - Efficiency analytics 
-- MAGIC - Audit analytics 
-- MAGIC - GDPR regulation
-- MAGIC - Service Level Objective analytics 
-- MAGIC - Data Quality analytics 
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Accessing your System tables With Unity Catalog 
-- MAGIC
-- MAGIC System Tables are available to customers who have Unity Catalog activated in at least one workspace. The data provided is collected from all workspaces in a Databricks account, regardless of the workspace's status with Unity Catalog. For example, if I have 10 workspaces and only one of them have Unity Catalog enabled then data is collected for all the workspaces and is made available via the single workspace in which Unity Catalog is active. 
-- MAGIC
-- MAGIC ### Enabling all system tables
-- MAGIC All systems tables are not enabled by default. As an account admin, you can review the [_enable_system_tables]($./_enable_system_tables) notebook to enable them all.

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## System Table Dashboard - Leverage AI with Lakehouse
-- MAGIC
-- MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/uc/system_tables/dashboard-governance-billing.png?raw=true" width="600px" style="float:right">
-- MAGIC
-- MAGIC We installed a Dashboard to track your billing and Unity Catalog usage leveraging the System tables.
-- MAGIC
-- MAGIC [Open the dashboard](/sql/dashboards/e23b9972-5574-4fbc-8ae7-b4e6c5ac7721) to review the informations available for you.<br/><br/>
-- MAGIC
-- MAGIC ### A note on Forecasting billing usage
-- MAGIC
-- MAGIC Please note that this dashboard forecasts your usage to predict your future spend and trigger potential alerts.
-- MAGIC
-- MAGIC To do so, we train multiple ML models leveraging `prophet` (the timeseries forecasting library). <br/>
-- MAGIC **Make sure you run the `01-billing-tables/02-forecast-billing-tables` notebook to generate the forecast data.** <br/>
-- MAGIC If you don't, data won't be available in the dashboard. `dbdemos` started a job in the background to initialize this data, but you can also directly run the notebook. 
-- MAGIC
-- MAGIC *For production-grade tracking, make sure you run your forecasting notebook as a job every day.*

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC ## Billing tables
-- MAGIC
-- MAGIC Billing table contains all the information required to track and analyze your consumption in DBU (Databricks Unit) and by extension $USD. 
-- MAGIC
-- MAGIC * To get started with the billing sytem tables, open the [01-billing-tables notebook]($./01-billing-tables/01-billing-tables-overview) to see how to explore your billing data
-- MAGIC * Leverage the Lakehouse capabilities to forecast your spend: open [02-forecast-billing-tables]($./01-billing-tables/02-forecast-billing-tables).
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Billing usage overview
select * from system.billing.usage limit 10

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC ## Audit Logs
-- MAGIC
-- MAGIC The audit log tables let you track and monitor all operations within your Lakehouse
-- MAGIC
-- MAGIC You can get information to understand your usage and diagnostic operations such as finding when a table was deleted, created...
-- MAGIC
-- MAGIC Open the [02-audit-log notebook]($./02-audit-logs-tables/02-audit-log) to explore your audit logs.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Audit log overview
select * from system.operational_data.audit_logs limit 10

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC ## Lineage
-- MAGIC
-- MAGIC Lineage tables let you track and monitor all lineage around your data assets.
-- MAGIC
-- MAGIC You can track at a Table and Column level to find dependencies and understand what your table is using including Workflows, SQL Queries, Dashboards, Notebooks...
-- MAGIC
-- MAGIC This information is available directly within the Data Explorer UI, but also saved as system table to simplify automated analysis.
-- MAGIC
-- MAGIC Open the [03-lineage notebook]($./03-lineage-tables/03-lineage) to explore your audit logs.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Lineage tables overview
SELECT * FROM system.lineage.table_lineage limit 10

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Conclusion
-- MAGIC
-- MAGIC With Unity Catalog System tables, you can easily monitor and exploit all your Lakehouse operations, from Audit Logs, Lineage up to Billing forecast.
-- MAGIC
-- MAGIC More system tables will be released soon, this demo will be updated accordingly, stay tuned!
