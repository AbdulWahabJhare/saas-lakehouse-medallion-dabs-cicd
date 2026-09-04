# Databricks notebook source
# 1. Define interactive UI widgets
dbutils.widgets.text("env", "dev", "1. Target Environment (dev/stage/prod)")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "2. Delta Write Mode")

# 2. Retrieve parameters
env = dbutils.widgets.get("env")
write_mode = dbutils.widgets.get("write_mode")
catalog_name = f"nexusmetrics_{env}"

# 3. Set Unity Catalog namespaces
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("USE SCHEMA silver")

print(f"✓ Environment Configured: Catalog = {catalog_name}, Schema = silver, Mode = {write_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC **1-Dimension Organizations — PII Masking & Schema Typing**

# COMMAND ----------

# MAGIC %md
# MAGIC What it does:
# MAGIC
# MAGIC Hashes company_name and assigned_ae using SHA-256 to create an anonymized pseudonym (e.g., ORG_MASKED_a4b9c1d2).
# MAGIC
# MAGIC Enforces timestamp data types on signup_date and churn_date.
# MAGIC
# MAGIC Computes a boolean is_active flag.
# MAGIC
# MAGIC Deduplicates records on the primary key org_id.
# MAGIC
# MAGIC Why we do it: Compliance with GDPR, HIPAA, and SOC-2 requires masking PII/company identities before exposing data to general business analysts or downstream data marts.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

print("Transforming: silver.dim_organizations...")

df_orgs_bronze = spark.table(f"{catalog_name}.bronze.organizations")

df_orgs_silver = df_orgs_bronze \
    .withColumn("company_name_masked", F.concat(F.lit("ORG_MASKED_"), F.substring(F.sha2(F.col("company_name"), 256), 1, 8))) \
    .withColumn("assigned_ae_masked", F.concat(F.lit("AE_"), F.substring(F.sha2(F.col("assigned_ae"), 256), 1, 6))) \
    .withColumn("signup_date", F.col("signup_date").cast(TimestampType())) \
    .withColumn("churn_date", F.col("churn_date").cast(TimestampType())) \
    .withColumn("is_active", F.when(F.col("is_churned") == 0, True).otherwise(False)) \
    .dropDuplicates(["org_id"])

(
    df_orgs_silver.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.silver.dim_organizations")
)

print(f"✓ Saved: silver.dim_organizations ({df_orgs_silver.count():,} rows)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from silver.dim_organizations limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC **2-Fact Subscription Billing — Financial Validation & ARR Derivation**

# COMMAND ----------

# MAGIC %md
# MAGIC What it does:
# MAGIC
# MAGIC Applies an integrity filter (mrr_amount >= 0) to prevent corrupted negative invoices.
# MAGIC
# MAGIC Derives annualized recurring revenue (annualized_run_rate = mrr_amount * 12).
# MAGIC
# MAGIC Enforces uniqueness on invoice_id.
# MAGIC
# MAGIC Why we do it: Financial reporting requires strict data assertions. Computing annualized run-rate at the Silver layer avoids repeated, compute-heavy recalculations in every BI dashboard.

# COMMAND ----------

print("Transforming: silver.fct_subscription_billing...")

df_billing_bronze = spark.table(f"{catalog_name}.bronze.billing")

df_billing_silver = df_billing_bronze \
    .filter(F.col("mrr_amount") >= 0) \
    .withColumn("invoice_date", F.col("invoice_date").cast(TimestampType())) \
    .withColumn("annualized_run_rate", F.round(F.col("mrr_amount") * 12, 2)) \
    .dropDuplicates(["invoice_id"])

(
    df_billing_silver.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.silver.fct_subscription_billing")
)

print(f"✓ Saved: silver.fct_subscription_billing ({df_billing_silver.count():,} rows)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from silver.fct_subscription_billing limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC **3-Fact Product Events — 10M+ Telemetry Scrubbing & Liquid Clustering**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC What it does:Drops invalid records where latency is non-positive.Adds categorical flags for quick indexing: is_error ($4\text{xx}/5\text{xx}$ codes), is_rate_limited (429), and is_slow_query ($>500\text{ms}$).Applies Liquid Clustering (clusterBy("org_id", "event_type")).
# MAGIC
# MAGIC Why we do it: Scanning 10M+ rows for every dashboard refresh is slow and expensive. Liquid clustering physically colocates data by customer and event type on disk, dramatically reducing file-skipping I/O and query latency.

# COMMAND ----------

print("Transforming: silver.fct_product_events (10M+ rows)...")

df_telemetry_bronze = spark.table(f"{catalog_name}.bronze.telemetry")

df_telemetry_silver = df_telemetry_bronze \
    .filter(F.col("latency_ms") > 0) \
    .withColumn("timestamp", F.col("timestamp").cast(TimestampType())) \
    .withColumn("is_error", F.when(F.col("http_status") >= 400, 1).otherwise(0)) \
    .withColumn("is_rate_limited", F.when(F.col("http_status") == 429, 1).otherwise(0)) \
    .withColumn("is_slow_query", F.when(F.col("latency_ms") > 500, 1).otherwise(0)) \
    .dropDuplicates(["event_id"])

(
    df_telemetry_silver.write
    .format("delta")
    .mode(write_mode)
    .clusterBy("org_id", "event_type")
    .saveAsTable(f"{catalog_name}.silver.fct_product_events")
)

print(f"✓ Saved: silver.fct_product_events (Optimized with Liquid Clustering)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from silver.fct_product_events limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC **4-CRM Deals, Support Tickets & CRO Experimentation Events**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC What it does:Enforces valid CSAT bounds ($1\text{--}5$) and flags dissatisfied clients (is_negative_sentiment = 1).Parses numeric step_number ($1\text{--}5$) from text strings in CRO events so funnel drop-off order is strictly sequential.
# MAGIC
# MAGIC Why we do it: Pre-classifying sentiment and structuring funnel steps makes joining these tables for machine learning feature stores and statistical analysis seamless.

# COMMAND ----------

from pyspark.sql.types import IntegerType

print("Transforming: CRM Deals, Support Tickets, and CRO Funnel Events...")

# 1. CRM Deals
df_deals_silver = spark.table(f"{catalog_name}.bronze.crm_deals") \
    .dropDuplicates(["deal_id"])

(
    df_deals_silver.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.silver.fct_crm_deals")
)

# 2. CRM Support Tickets (CSAT validation & sentiment classification)
df_tickets_silver = spark.table(f"{catalog_name}.bronze.crm_tickets") \
    .filter((F.col("csat_score") >= 1) & (F.col("csat_score") <= 5)) \
    .withColumn("created_at", F.col("created_at").cast(TimestampType())) \
    .withColumn("is_negative_sentiment", F.when(F.col("csat_score") <= 2, 1).otherwise(0)) \
    .dropDuplicates(["ticket_id"])

(
    df_tickets_silver.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.silver.fct_crm_support_tickets")
)

# 3. CRO Funnel Events (Extract step numbers for progression analysis)
df_cro_silver = spark.table(f"{catalog_name}.bronze.cro_funnel") \
    .withColumn("timestamp", F.col("timestamp").cast(TimestampType())) \
    .withColumn("step_number", F.substring(F.col("step_name"), 1, 1).cast(IntegerType())) \
    .dropDuplicates(["visitor_id", "step_name"])

(
    df_cro_silver.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.silver.fct_cro_funnel_events")
)

print("✓ Saved: silver.fct_crm_deals, silver.fct_crm_support_tickets, silver.fct_cro_funnel_events")

# COMMAND ----------

# MAGIC %md
# MAGIC **End-to-End Silver Layer Verification Audit**

# COMMAND ----------

silver_tables = [
    f"{catalog_name}.silver.dim_organizations",
    f"{catalog_name}.silver.fct_subscription_billing",
    f"{catalog_name}.silver.fct_product_events",
    f"{catalog_name}.silver.fct_crm_deals",
    f"{catalog_name}.silver.fct_crm_support_tickets",
    f"{catalog_name}.silver.fct_cro_funnel_events"
]

print(f"\n{'TABLE NAME':<55} | {'ROW COUNT':>15}")
print("-" * 73)
for t in silver_tables:
    cnt = spark.table(t).count()
    print(f"{t:<55} | {cnt:>15,}")
print("-" * 73)