# Risk Factor by Zip Code in the US

This repository provides documentation and notebooks
for a data pipline in Databricks to create a ML model for determining the risk factor for each zip code in the US.


Table of Contents
=================

   * [Databricks Risk Factor by Zip in the US](#databricks-risk-factor-by-zip-in-the-us)
      * [Table of Contents](#table-of-contents)
      * [Project File Structure](#project-file-structure)
      * [Overview](#overview)
         * [Gathering Datasets](#gathering-datasets)
         * [Unity Catalog Setup](#local-steps)
         * [Data Ingestion](#data-ingestion)
         * [Data Transformation](#data-transformation)
         * [ML Model for Determining Risk Factor](#ml-model-for-determining-risk-factor)
      * [Notebooks](#notebooks)
    
## Project File Structure
```
. 
├── .bronze
│   ├── azure_extract
|   ├── combined_census
|   └── other_refined_data
├── .silver
│   ├── census_silver_notebook
|   ├── happiness_index_regression
|   └── risk_factor
├── .gold
|   └── census_gold_notebook
└── README.md
```

Some descriptions regarding files:
- `azure_extract` - Handles data ingestion by gathering datasets from Azure Storage Container.
- `combined_census` - Combines Census datasets, which were seperate datasets for each year from 2011-2018.
- `other_refined_data` - Pulls in other relevant datasets that had data per zip code.
- `census_silver_notebook` - Performs queries on the combined Census data to extract relevant information that could contribute to risk factor and provides visualizations.
- `happiness_index_regression` - Creates a linear regression using arbitrary weights assigned to different features from census data that could potentially impact happiness. The result is a synthetic happines index.
- `risk_factor` - Takes various features from the Census data and uses the Elbow Method to determine the optimal number of clusters (risk factor levels in this case), and uses a KMeans model with that optimal number of clusters to put each zip code into its appropraite risk factor cluster [low risk, moderate risk, high risk].
- `census_gold_notebook` - Performs queries on the Census data by zip code with the risk factor for each zip included.

## Overview

> **_NOTE:_**  
At the time of writing, [ML Models in Unity Catalog](https://learn.microsoft.com/en-us/azure/databricks/mlflow/models-in-uc-example) are in Public Peview.
In this project, we are using [Databricks Runtime 13.2 ML](https://docs.databricks.com/en/release-notes/runtime/13.2ml.html).

### Gathering datasets
We sourced our Census data from Google Big Query's public datasets:
- [US Census Data on Google Big Query](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=census_bureau_usa&page=dataset&project=ardent-fusion-394216&ws=!1m4!1m3!3m2!1sbigquery-public-data!2scensus_bureau_acs)

The datasets were separated by year from 2011-2018. Thus, we had to do some data transformation to combine these datasets into one table, which can be found [here](#data-transformation).

### Unity Catalog Setup

- Setup a Unity Catalog metastore for easier data governance.
- Followed this easy setup guide on YouTube: [Unity Catalog setup for Azure Databricks](https://www.youtube.com/watch?v=-RwzDRVgjLc), as well as the official [Azure Documentation](https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/get-started).


### Data ingestion

- Downloaded Census Data from Google Big Query as parquet.
- Uploaded parquet files to this [Azure Storage Container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fd402e5e9-4b24-4f30-933d-a097415411b6%2FresourceGroups%2FWOLCM01%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fraw001/path/raw/etag/%220x8DB9468600B6E20%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None).
- Created azure_extract notebook to gather all parquet files from Azure Storage Container and save them as Delta Tables in our bronze tier data.

### Data transformation

- Created combined_census notebook and pulled in all delta tables for Census data for separate years (2011-2018) from bronze data tier.
- Added primary key to each table formatted: `zipcode_year`.
- Performed a union one table at a time, checking for missing columns to ensure the same schema for all tables.
    * If a table was missing columns, they would be added and filled with null values
- The combined delta table was then saved to the silver data tier.

### ML model for determining risk factor
- Using the combined Census delta table (main.silver.census_2011_to_2018), features that could impact the risk factor of a particular zip code were extracted using a Spark SQL query and converted into a Pandas dataframe.
- For any missing values, mean imputation was used to fill in those gaps.
- The features were then normalized and used in the Elbow Method to determine the optimal number of risk factor clusters.
- After plotting and looking at the Elbow Method graph, it appeared that 3-4 clusters was the optimal number for risk factor clusters.
- Thus, 3 clusters were used for the KMeans model implementation: [`Low Risk`, `Moderate Risk`, `High Risk`]
- The Risk Factor Distribution by Cluster (or Risk Level) is then plotted using a bar graph.
- Finally, a few notebook cells are used to show the risk factor and features from 2011-2018 for one of the worst neighborhoods in Chicago, one of the lowest poverty neighborhoods in the US, and finally for any zip code typed into the 'zip' Databricks widget at the top of the notebook.

 
## Notebooks
