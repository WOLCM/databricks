# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG main;
# MAGIC USE main.silver;

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Silver Tests") \
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

spark.sql("""CREATE or REPLACE TEMPORARY VIEW core_view AS
SELECT 
    *
FROM 
    silver.census_2011_to_2018
where geo_id = '{zip_code}'
""".format(zip_code = zip_))

housing_df = spark.sql('''
                        select geo_id, year, housing_units, occupied_housing_units, housing_units_renter_occupied, mobile_homes, housing_built_1939_or_earlier, housing_built_2000_to_2004, housing_built_2005_or_later,
                        vacant_housing_units, vacant_housing_units_for_rent, vacant_housing_units_for_sale
                        from core_view
                        order by primary_key asc
                        ''')

display(housing_df)


# COMMAND ----------

age_and_gender_df = spark.sql('''
                              select geo_id, year,
                                (male_under_5 + male_5_to_9 + male_10_to_14 + male_15_to_17) as male_under_18,
                                (male_18_to_19 + male_20 + male_21 + male_22_to_24 + male_25_to_29) as male_over_18_under_30,
                                (male_30_to_34 + male_35_to_39) as male_over_30_under_40,
                                (male_40_to_44 + male_45_to_49) as male_over_40_under_50,
                                (male_50_to_54 + male_55_to_59) as male_over_50_under_60,
                                (male_60_61 + male_62_64 + male_65_to_66 + male_67_to_69) as male_over_60_under_70,
                                (male_70_to_74 + male_75_to_79) as male_over_70_under_80,
                                (male_80_to_84 + male_85_and_over) as male_80_and_over,

                                (female_under_5 + female_5_to_9 + female_10_to_14 + female_15_to_17) as female_under_18,
                                (female_18_to_19 + female_20 + female_21 + female_22_to_24 + female_25_to_29) as female_over_18_under_30,
                                (female_30_to_34 + female_35_to_39) as female_over_30_under_40,
                                (female_40_to_44 + female_45_to_49) as female_over_40_under_50,
                                (female_50_to_54 + female_55_to_59) as female_over_50_under_60,
                                (female_60_to_61 + female_62_to_64 + female_65_to_66 + female_67_to_69) as female_over_60_under_70,
                                (female_70_to_74 + female_75_to_79) as female_over_70_under_80,
                                (female_80_to_84 + female_85_and_over) as female_80_and_over

                                from core_view
                                order by primary_key asc
                                ''')

display(age_and_gender_df)

# COMMAND ----------

pop_and_race_df = spark.sql('''
                          select geo_id, year, total_pop, white_pop, black_pop, asian_pop, hispanic_pop, amerindian_pop, other_race_pop
                          from core_view
                          order by primary_key asc
                          ''')

display(pop_and_race_df)

# COMMAND ----------

commute_df = spark.sql('''
                        select geo_id, year,
                            commute_less_10_mins,
                            (commute_10_14_mins + commute_15_19_mins + commute_20_24_mins + commute_25_29_mins) as commute_over_10_less_30_mins,
                            (commute_30_34_mins + commute_35_39_mins + commute_40_44_mins + commute_45_59_mins) as commute_over_30_less_1_hr,
                            commute_60_more_mins,
                            commute_90_more_mins,

                            commuters_by_public_transportation, commuters_by_subway_or_elevated, commuters_drove_alone, commuters_by_carpool, commuters_by_car_truck_van, commuters_by_bus

                        from core_view
                        order by primary_key asc
                        ''')
display(commute_df)

# COMMAND ----------

all_df = spark.sql('''
                        select *
                        from core_view
                        order by primary_key asc
                        ''')

display(all_df)
