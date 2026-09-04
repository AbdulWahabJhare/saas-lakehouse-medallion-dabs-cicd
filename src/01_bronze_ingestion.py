# Databricks notebook source
# MAGIC %md 
# MAGIC ### Data Ingestion

# COMMAND ----------

# 1. Clear existing widgets (optional, useful when resetting)
# dbutils.widgets.removeAll()

# 2. Define interactive widgets
dbutils.widgets.text("s3_bucket", "nexusmetrics-lakehouse-raw", "1. S3 Bucket Name")
dbutils.widgets.text("env", "dev", "2. Environment (dev/stage/prod)")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "3. Write Mode")
dbutils.widgets.text("raw_folder_prefix", "raw", "4. Raw S3 Folder Prefix")

# 3. Retrieve values from UI
s3_bucket_name = dbutils.widgets.get("s3_bucket")
env = dbutils.widgets.get("env")
write_mode = dbutils.widgets.get("write_mode")
folder_prefix = dbutils.widgets.get("raw_folder_prefix")

# 4. Construct dynamic paths and namespaces
catalog_name = f"nexusmetrics_{env}"
s3_base = f"s3://{s3_bucket_name}/{folder_prefix}"

print(f"Target Unity Catalog : {catalog_name}")
print(f"Target Schema        : {catalog_name}.bronze")
print(f"S3 Base Path         : {s3_base}")
print(f"Delta Write Mode     : {write_mode}")

# COMMAND ----------

print(f'spark version : {spark.version}')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create database/schema for your Bronze layer
# MAGIC CREATE CATALOG IF NOT EXISTS nexusmetrics_dev;
# MAGIC USE CATALOG nexusmetrics_dev;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS gold;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading  from S3 and Write to Bronze Delta Tables

# COMMAND ----------

BUCKET_NAME = "nexusmetrics-lakehouse-raw"
s3_base = f"s3://{BUCKET_NAME}"

tables = {
    "organizations": f"{s3_base}/dimensions/dim_organizations.parquet",
    "billing": f"{s3_base}/billing/fct_subscription_billing.parquet",
    "telemetry": f"{s3_base}/telemetry/fct_product_events.parquet",
    "crm_deals": f"{s3_base}/crm/fct_crm_deals.parquet",
    "crm_tickets": f"{s3_base}/crm/fct_crm_support_tickets.parquet",
    "cro_funnel": f"{s3_base}/cro/fct_cro_funnel_events.parquet"
}

for table_name, s3_path in tables.items():
    print(f"Ingesting {table_name} from {s3_path}...")
    df = spark.read.parquet(s3_path)
    df.write.format("delta").mode("overwrite").saveAsTable(f"nexusmetrics_dev.bronze.{table_name}")
    print(f"✓ Created bronze.{table_name}")

# COMMAND ----------

bronze_tables = [
    "nexusmetrics_dev.bronze.organizations",
    "nexusmetrics_dev.bronze.billing",
    "nexusmetrics_dev.bronze.telemetry",
    "nexusmetrics_dev.bronze.crm_deals",
    "nexusmetrics_dev.bronze.crm_tickets",
    "nexusmetrics_dev.bronze.cro_funnel"
]

print(f"{'TABLE NAME':<40} | {'ROW COUNT':>15}")
print("-" * 58)

for table_name in bronze_tables:
    count = spark.table(table_name).count()
    short_name = table_name.split(".")[-1]
    print(f"{short_name:<40} | {count:>15,}")

print("-" * 58)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM nexusmetrics_dev.bronze.organizations LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from nexusmetrics_dev.bronze.telemetry

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from nexusmetrics_dev.bronze.cro_funnel