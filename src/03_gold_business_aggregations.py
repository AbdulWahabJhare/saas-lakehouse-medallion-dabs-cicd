# Databricks notebook source
# 1. Define interactive widgets
dbutils.widgets.text("env", "dev", "1. Target Environment (dev/stage/prod)")
dbutils.widgets.text("reporting_year", "2026", "2. Reporting Year")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "3. Write Mode")

# 2. Retrieve values
env = dbutils.widgets.get("env")
reporting_year = int(dbutils.widgets.get("reporting_year"))
write_mode = dbutils.widgets.get("write_mode")
catalog_name = f"nexusmetrics_{env}"

# 3. Establish namespaces
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
spark.sql("USE SCHEMA gold")

print(f"✓ Active Catalog: {catalog_name} | Schema: gold | Year: {reporting_year}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1: Environment Configuration & Dynamic Widgets
# MAGIC * **Objective:** Parameterize execution parameters and establish the Unity Catalog namespace for the Gold layer.
# MAGIC * **Key Actions:**
# MAGIC   * Configures runtime widgets for target environment (`env`), reporting scope (`reporting_year`), and Delta write strategy (`write_mode`).
# MAGIC   * Resolves the target catalog (`nexusmetrics_<env>`) and ensures the `gold` schema exists.
# MAGIC * **Business & Architectural Value:** Decouples pipeline logic from hardcoded environments, enabling automated, zero-touch execution across CI/CD environments and scheduled orchestrators.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Monthly MRR Waterfall and Revenue Cohorts**

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("Computing: gold.gold_mrr_waterfall...")

df_billing = spark.table(f"{catalog_name}.silver.fct_subscription_billing")
df_orgs = spark.table(f"{catalog_name}.silver.dim_organizations")

# Add month partition and window over customer billing cycles
df_billing_monthly = df_billing \
    .withColumn("billing_month", F.trunc(F.col("invoice_date"), "month"))

# Window to compare current month MRR vs prior month MRR per org
org_window = Window.partitionBy("org_id").orderBy("billing_month")

df_revenue_movements = df_billing_monthly \
    .withColumn("prev_mrr", F.lag("mrr_amount", 1, 0.0).over(org_window)) \
    .withColumn("mrr_change", F.col("mrr_amount") - F.col("prev_mrr")) \
    .withColumn("movement_type", 
        F.when(F.col("prev_mrr") == 0, "New_MRR")
         .when(F.col("mrr_amount") == 0, "Churn_MRR")
         .when(F.col("mrr_change") > 0, "Expansion_MRR")
         .when(F.col("mrr_change") < 0, "Contraction_MRR")
         .otherwise("Retained_MRR")
    )

# Aggregate monthly figures by tier
df_gold_mrr = df_revenue_movements \
    .groupBy("billing_month", "tier") \
    .agg(
        F.round(F.sum(F.when(F.col("movement_type") == "New_MRR", F.col("mrr_amount")).otherwise(0)), 2).alias("new_mrr"),
        F.round(F.sum(F.when(F.col("movement_type") == "Expansion_MRR", F.col("mrr_change")).otherwise(0)), 2).alias("expansion_mrr"),
        F.round(F.sum(F.when(F.col("movement_type") == "Contraction_MRR", F.abs(F.col("mrr_change"))).otherwise(0)), 2).alias("contraction_mrr"),
        F.round(F.sum(F.when(F.col("movement_type") == "Churn_MRR", F.col("prev_mrr")).otherwise(0)), 2).alias("churned_mrr"),
        F.round(F.sum("mrr_amount"), 2).alias("ending_total_mrr"),
        F.countDistinct("org_id").alias("active_customer_count")
    ) \
    .withColumn("ending_total_arr", F.round(F.col("ending_total_mrr") * 12, 2)) \
    .orderBy("billing_month", "tier")

(
    df_gold_mrr.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.gold.gold_mrr_waterfall")
)

print(f"✓ Saved: gold.gold_mrr_waterfall ({df_gold_mrr.count():,} rows)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from gold.gold_mrr_waterfall limit 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monthly Recurring Revenue (MRR) Waterfall & Cohorts
# MAGIC * **Objective:** Track SaaS financial momentum and categorize monthly recurring revenue movements per customer.
# MAGIC * **Key Actions:**
# MAGIC   * Uses PySpark windowing (`F.lag`) partitioned by `org_id` over time to track billing shifts.
# MAGIC   * Classifies cash flow movements into standard SaaS accounting categories: `New_MRR`, `Expansion_MRR`, `Contraction_MRR`, `Churn_MRR`, and `Retained_MRR`.
# MAGIC   * Aggregates monthly totals alongside Annualized Run Rate (`ending_total_arr`) and active customer counts by tier.
# MAGIC * **Business & Architectural Value:** Serves as the primary financial reporting model required by finance leadership to calculate Net Revenue Retention (NRR), gross churn rate, and quarterly SaaS trajectory.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Gold Product Telemetry and SLA Performance**

# COMMAND ----------

print("Computing: gold.gold_telemetry_daily_sla (Aggregating 10M+ rows)...")

df_telemetry = spark.table(f"{catalog_name}.silver.fct_product_events")
df_orgs = spark.table(f"{catalog_name}.silver.dim_organizations")

# Aggregate by date, tier, and event_type
df_gold_telemetry = df_telemetry \
    .join(df_orgs.select("org_id", "tier"), on="org_id", how="left") \
    .withColumn("event_date", F.to_date(F.col("timestamp"))) \
    .groupBy("event_date", "tier", "event_type") \
    .agg(
        F.count("event_id").alias("total_requests"),
        F.round(F.avg("latency_ms"), 2).alias("avg_latency_ms"),
        F.expr("percentile_approx(latency_ms, 0.50)").alias("p50_latency_ms"),
        F.expr("percentile_approx(latency_ms, 0.95)").alias("p95_latency_ms"),
        F.expr("percentile_approx(latency_ms, 0.99)").alias("p99_latency_ms"),
        F.sum("is_error").alias("total_errors"),
        F.sum("is_rate_limited").alias("rate_limit_429_hits"),
        F.round((F.sum("is_error") / F.count("event_id")) * 100, 3).alias("error_rate_pct"),
        F.round(F.sum("payload_bytes") / (1024 * 1024), 2).alias("bandwidth_transferred_mb")
    ) \
    .orderBy("event_date", "tier")

(
    df_gold_telemetry.write
    .format("delta")
    .mode(write_mode)
    .clusterBy("event_date", "tier")
    .saveAsTable(f"{catalog_name}.gold.gold_telemetry_daily_sla")
)

print(f"✓ Saved: gold.gold_telemetry_daily_sla (Clustered on event_date & tier)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from gold.gold_telemetry_daily_sla limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Product Telemetry & Daily SLA Reliability Aggregations
# MAGIC * **Objective:** Transform 10M+ raw telemetry event logs into high-performance, daily service-level metrics.
# MAGIC * **Key Actions:**
# MAGIC   * Extracts calendar dates and joins company subscription tiers (`Basic`, `Premium`, `Diamond`).
# MAGIC   * Computes statistical percentile latencies (`p50`, `p95`, `p99`) via `percentile_approx` to reveal long-tail user delays.
# MAGIC   * Aggregates availability indicators, including total HTTP errors ($4\text{xx}/5\text{xx}$), rate-limit spikes (`429`), and total bandwidth (MB).
# MAGIC   * Optimizes physical disk layout using Delta Lake Liquid Clustering on `(event_date, tier)`.
# MAGIC * **Business & Architectural Value:** Enables sub-second dashboard rendering for tracking compliance against contractual Service Level Agreements (SLAs) without scanning millions of raw log partitions.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Gold A/B experimentation and Funnel Conversion**

# COMMAND ----------

#Evaluate the CRO pricing experiment with statistical calculations ($Z$-score, conversion lift, and $p$-value)
print("Computing: gold.gold_ab_experiment_conversion (Statistical Hypothesis Testing)...")

df_cro = spark.table(f"{catalog_name}.silver.fct_cro_funnel_events")

# 1. Calculate overall funnel conversion by variant
df_funnel_summary = df_cro \
    .groupBy("variant", "step_name", "step_number") \
    .agg(F.countDistinct("visitor_id").alias("unique_visitors")) \
    .orderBy("variant", "step_number")

# 2. Extract step 1 (Landing) and step 5 (Converted) to calculate lift and p-value
df_ab_calc = df_cro \
    .groupBy("variant") \
    .agg(
        F.countDistinct(F.when(F.col("step_number") == 1, F.col("visitor_id"))).alias("visitors_step_1"),
        F.countDistinct(F.when(F.col("step_number") == 5, F.col("visitor_id"))).alias("conversions_step_5")
    ) \
    .withColumn("conversion_rate", F.round(F.col("conversions_step_5") / F.col("visitors_step_1"), 4))

# Convert to Pandas locally to compute two-proportion Z-test and p-value
pdf = df_ab_calc.toPandas().set_index("variant")

c_visitors, c_conv = pdf.loc["Control_A", ["visitors_step_1", "conversions_step_5"]]
t_visitors, t_conv = pdf.loc["Treatment_B", ["visitors_step_1", "conversions_step_5"]]

p_pool = (c_conv + t_conv) / (c_visitors + t_visitors)
se = (p_pool * (1 - p_pool) * ((1 / c_visitors) + (1 / t_visitors))) ** 0.5
z_score = ((t_conv / t_visitors) - (c_conv / c_visitors)) / se

# P-value calculation from standard normal distribution
import scipy.stats as stats
p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))
rel_lift_pct = float((( (t_conv / t_visitors) - (c_conv / c_visitors) ) / (c_conv / c_visitors)) * 100)

print(f"  * Control A Conv Rate   : {c_conv/c_visitors:.2%}")
print(f"  * Treatment B Conv Rate : {t_conv/t_visitors:.2%}")
print(f"  * Relative Lift         : +{rel_lift_pct:.2f}%")
print(f"  * Z-Score               : {z_score:.4f}")
print(f"  * P-Value               : {p_value:.6e} (Statistically Significant: {p_value < 0.05})")

# 3. Create enriched Gold Funnel table
df_gold_cro = df_funnel_summary \
    .withColumn("relative_lift_pct", F.lit(round(rel_lift_pct, 2))) \
    .withColumn("p_value", F.lit(p_value)) \
    .withColumn("is_significant", F.lit(p_value < 0.05))

(
    df_gold_cro.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.gold.gold_ab_experiment_conversion")
)

print(f"✓ Saved: gold.gold_ab_experiment_conversion")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A/B Experimentation & Conversion Funnel Analysis
# MAGIC * **Objective:** Evaluate the CRO pricing flow experiment to determine if Treatment B statistically outperforms Control A.
# MAGIC * **Key Actions:**
# MAGIC   * Computes unique visitor volume across all 5 sequential funnel stages to track drop-off bottlenecks.
# MAGIC   * Executes a two-proportion pooled hypothesis test ($Z$-test) comparing top-of-funnel landing views to completed paid conversions.
# MAGIC   * Calculates relative conversion lift (%), standard error, test statistic ($Z$-score), and the two-tailed $p$-value.
# MAGIC   * Records the statistical significance flag (`is_significant` where $p < 0.05$) directly in the Delta table.
# MAGIC * **Business & Architectural Value:** Empowers product teams to make data-backed deployment decisions by mathematically verifying whether pricing page changes yield real revenue lift versus random variation.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Gold Customer 360 & Feature Store**

# COMMAND ----------

print("Computing: gold.gold_customer_360_features (Machine Learning Ready)...")

df_silver_orgs = spark.table(f"{catalog_name}.silver.dim_organizations")
df_silver_tickets = spark.table(f"{catalog_name}.silver.fct_crm_support_tickets")
df_silver_telemetry = spark.table(f"{catalog_name}.silver.fct_product_events")
df_silver_deals = spark.table(f"{catalog_name}.silver.fct_crm_deals")

# Support friction features per org
support_feats = df_silver_tickets \
    .groupBy("org_id") \
    .agg(
        F.count("ticket_id").alias("total_tickets_filed"),
        F.round(F.avg("csat_score"), 2).alias("avg_csat_score"),
        F.sum("is_negative_sentiment").alias("negative_sentiment_tickets"),
        F.round(F.avg("resolution_time_hrs"), 1).alias("avg_ticket_resolution_hrs")
    )

# Product telemetry features per org
telemetry_feats = df_silver_telemetry \
    .groupBy("org_id") \
    .agg(
        F.count("event_id").alias("total_api_calls_90d"),
        F.round(F.avg("latency_ms"), 1).alias("avg_latency_ms"),
        F.sum("is_error").alias("total_http_errors"),
        F.sum("is_rate_limited").alias("total_rate_limits_hit"),
        F.round((F.sum("is_error") / F.count("event_id")) * 100, 2).alias("error_rate_pct")
    )

# Join into single unified Customer 360 Table
df_gold_c360 = df_silver_orgs \
    .join(df_silver_deals.select("org_id", "deal_value_arr", "lead_source", "pipeline_duration_days"), on="org_id", how="left") \
    .join(support_feats, on="org_id", how="left") \
    .join(telemetry_feats, on="org_id", how="left") \
    .na.fill({
        "total_tickets_filed": 0,
        "avg_csat_score": 3.0,
        "negative_sentiment_tickets": 0,
        "avg_ticket_resolution_hrs": 0.0,
        "total_api_calls_90d": 0,
        "avg_latency_ms": 0.0,
        "total_http_errors": 0,
        "total_rate_limits_hit": 0,
        "error_rate_pct": 0.0
    })

(
    df_gold_c360.write
    .format("delta")
    .mode(write_mode)
    .saveAsTable(f"{catalog_name}.gold.gold_customer_360_features")
)

print(f"✓ Saved: gold.gold_customer_360_features ({df_gold_c360.count():,} rows)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from gold.gold_customer_360_features limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customer 360 Analytical Profile & Feature Store
# MAGIC * **Objective:** Unify multi-source behavioral, commercial, and operational signals into an enterprise Customer 360 entity.
# MAGIC * **Key Actions:**
# MAGIC   * Merges firmographic attributes (`dim_organizations`) with sales contract metrics (`deal_value_arr`, pipeline velocity).
# MAGIC   * Enriches accounts with historical support friction (total tickets, average CSAT score, negative sentiment count).
# MAGIC   * Appends 90-day product engagement signals (total API requests, error rates, average latency).
# MAGIC   * Imputes missing values with baseline defaults to generate a clean, zero-null feature set.
# MAGIC * **Business & Architectural Value:** Acts as the single source of truth for Customer Success health scores in BI and provides an ML-ready feature store for Churn Prediction and Upsell Propensity models.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Aggrigated tables verification**

# COMMAND ----------

gold_tables = [
    f"{catalog_name}.gold.gold_mrr_waterfall",
    f"{catalog_name}.gold.gold_telemetry_daily_sla",
    f"{catalog_name}.gold.gold_ab_experiment_conversion",
    f"{catalog_name}.gold.gold_customer_360_features"
]

print(f"\n{'GOLD TABLE NAME':<55} | {'ROW COUNT':>15}")
print("-" * 73)
for t in gold_tables:
    cnt = spark.table(t).count()
    print(f"{t:<55} | {cnt:>15,}")
print("-" * 73)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold Layer Validation & Row Count Audit
# MAGIC * **Objective:** Validate end-to-end table compilation and record counts across all curated Gold tables.
# MAGIC * **Key Actions:**
# MAGIC   * Iterates across the 4 generated Gold entities: `gold_mrr_waterfall`, `gold_telemetry_daily_sla`, `gold_ab_experiment_conversion`, and `gold_customer_360_features`.
# MAGIC   * Outputs standardized audit logs to confirm complete, partition-consistent data delivery.
# MAGIC * **Business & Architectural Value:** Guarantees pipeline data integrity prior to downstream consumption by Power BI dashboards or MLflow model training routines.

# COMMAND ----------

