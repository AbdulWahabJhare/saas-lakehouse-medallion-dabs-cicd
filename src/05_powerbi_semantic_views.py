# Databricks notebook source
# MAGIC %md
# MAGIC **Dynamic Parametere Widgets & Semantic View Setup**

# COMMAND ----------

# 1. Define interactive widgets
dbutils.widgets.text("env", "dev", "1. Target Environment (dev/stage/prod)")
dbutils.widgets.text("reporting_currency", "USD", "2. Currency Symbol")

# 2. Retrieve values
env = dbutils.widgets.get("env")
currency = dbutils.widgets.get("reporting_currency")
catalog_name = f"nexusmetrics_{env}"

# 3. Create semantic consumption schema inside Unity Catalog
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("CREATE SCHEMA IF NOT EXISTS semantic_views")
spark.sql("USE SCHEMA semantic_views")

print(f"✓ Target Catalog: {catalog_name}")
print(f"✓ Semantic Schema Initialized: {catalog_name}.semantic_views")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1: Semantic Layer Initialization & Namespace Configuration
# MAGIC * **Objective:** Establish the dynamic environment parameters and dedicated consumption schema for Power BI views.
# MAGIC * **Key Actions:**
# MAGIC   * Configures runtime widgets for environment targeting (`env`) and reporting currency (`reporting_currency`).
# MAGIC   * Creates and switches context to the `semantic_views` schema within Unity Catalog.
# MAGIC * **Business & Architectural Value:** Isolates business consumption views from underlying physical Gold tables, ensuring analysts query clean logical models without risking schema locks or altering underlying data.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Customer 360 & AI Churn Semantic Viewe (vw_dim_customers)**

# COMMAND ----------

print("Creating semantic view: vw_dim_customers...")

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog_name}.semantic_views.vw_dim_customers AS
SELECT 
    c.org_id AS customer_id,
    c.company_name_masked AS company_display_name,
    c.tier AS subscription_tier,
    c.industry,
    c.signup_date,
    c.churn_date,
    c.is_active AS is_active_customer,
    c.deal_value_arr AS annualized_contract_value,
    c.lead_source AS acquisition_channel,
    c.pipeline_duration_days AS sales_cycle_days,
    c.total_tickets_filed,
    c.avg_csat_score,
    c.negative_sentiment_tickets,
    c.total_api_calls_90d,
    c.avg_latency_ms,
    c.error_rate_pct,
    -- ML Model Predictions
    p.churn_risk_score,
    p.risk_band AS churn_risk_tier,
    CASE 
        WHEN p.risk_band = 'High Risk' THEN 'Action Required - Immediate Outreach'
        WHEN p.risk_band = 'Medium Risk' THEN 'Monitor Engagement'
        ELSE 'Healthy Account'
    END AS customer_health_status
FROM {catalog_name}.gold.gold_customer_360_features c
LEFT JOIN {catalog_name}.gold.org_churn_predictions p
    ON c.org_id = p.org_id
""")

print("✓ Created: semantic_views.vw_dim_customers")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     customer_health_status,
# MAGIC     subscription_tier,
# MAGIC     COUNT(*) AS total_accounts,
# MAGIC     ROUND(AVG(churn_risk_score) * 100, 2) AS avg_churn_risk_pct,
# MAGIC     ROUND(SUM(annualized_contract_value), 2) AS total_arr_at_risk
# MAGIC FROM semantic_views.vw_dim_customers
# MAGIC WHERE is_active_customer = true
# MAGIC GROUP BY customer_health_status, subscription_tier
# MAGIC ORDER BY total_arr_at_risk DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 & 3: Dimension Customers & Churn Risk Semantic View
# MAGIC * **Objective:** Deliver a unified customer dimension combining demographic profiles, support sentiment, platform usage, and predictive churn risk scores.
# MAGIC * **Key Actions:**
# MAGIC   * Joins `gold_customer_360_features` with `org_churn_predictions`.
# MAGIC   * Formats column headers into standardized business terminology (`customer_id`, `subscription_tier`, `annualized_contract_value`).
# MAGIC   * Generates an operational triage flag (`customer_health_status`) to direct Customer Success workflows.
# MAGIC * **Business & Architectural Value:** Powers Customer 360 dashboards in Power BI, enabling account managers to identify at-risk enterprise ARR and prioritize high-value client interventions.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Fact MRR Waterfall View (vw_fact_mrr_waterfall)**

# COMMAND ----------

print("Creating semantic view: vw_fct_mrr_waterfall...")

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog_name}.semantic_views.vw_fct_mrr_waterfall AS
SELECT 
    billing_month AS transaction_month,
    tier AS subscription_tier,
    new_mrr AS new_mrr_amount,
    expansion_mrr AS expansion_mrr_amount,
    contraction_mrr AS contraction_mrr_amount,
    churned_mrr AS churn_mrr_amount,
    (new_mrr + expansion_mrr - contraction_mrr - churned_mrr) AS net_mrr_change,
    ending_total_mrr,
    ending_total_arr,
    active_customer_count
FROM {catalog_name}.gold.gold_mrr_waterfall
""")

print("✓ Created: semantic_views.vw_fct_mrr_waterfall")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     transaction_month,
# MAGIC     subscription_tier,
# MAGIC     ending_total_mrr,
# MAGIC     net_mrr_change,
# MAGIC     active_customer_count
# MAGIC FROM semantic_views.vw_fct_mrr_waterfall
# MAGIC ORDER BY transaction_month DESC, ending_total_mrr DESC
# MAGIC LIMIT 12;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 4 & 5: Fact MRR Waterfall Semantic View
# MAGIC * **Objective:** Expose standardized monthly financial recurring revenue mechanics for executive reporting.
# MAGIC * **Key Actions:**
# MAGIC   * Maps monthly billing rollups into distinct revenue components: New, Expansion, Contraction, and Churn.
# MAGIC   * Calculates `net_mrr_change` directly within the view to simplify downstream BI measure creation.
# MAGIC * **Business & Architectural Value:** Enables CFO and Board-level visual reporting for Net Revenue Retention (NRR) and Annual Recurring Revenue (ARR) growth curves without complex client-side calculations.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Fact Telemetry SlA & Fact CRO Funnel Views**

# COMMAND ----------

print("Creating semantic views: vw_fct_telemetry_sla and vw_fct_cro_funnel...")

# 1. Telemetry SLA View
spark.sql(f"""
CREATE OR REPLACE VIEW {catalog_name}.semantic_views.vw_fct_telemetry_sla AS
SELECT 
    event_date,
    tier AS subscription_tier,
    event_type,
    total_requests,
    avg_latency_ms,
    p50_latency_ms AS median_latency_ms,
    p95_latency_ms,
    p99_latency_ms,
    total_errors,
    rate_limit_429_hits,
    error_rate_pct,
    bandwidth_transferred_mb,
    CASE 
        WHEN p95_latency_ms <= 300.0 AND error_rate_pct < 0.05 THEN 'SLA Compliant'
        ELSE 'SLA Breach / At Risk'
    END AS sla_operational_status
FROM {catalog_name}.gold.gold_telemetry_daily_sla
""")

# 2. CRO Experimentation Funnel View
spark.sql(f"""
CREATE OR REPLACE VIEW {catalog_name}.semantic_views.vw_fct_cro_funnel AS
SELECT 
    variant AS test_variant,
    step_number,
    step_name AS funnel_stage_name,
    unique_visitors,
    relative_lift_pct,
    ROUND(p_value, 5) AS statistical_p_value,
    CASE 
        WHEN is_significant = true THEN 'Statistically Significant (p < 0.05)'
        ELSE 'Not Significant (Inconclusive)'
    END AS test_significance_status
FROM {catalog_name}.gold.gold_ab_experiment_conversion
""")

print("✓ Created: semantic_views.vw_fct_telemetry_sla")
print("✓ Created: semantic_views.vw_fct_cro_funnel")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     test_variant,
# MAGIC     step_number,
# MAGIC     funnel_stage_name,
# MAGIC     unique_visitors,
# MAGIC     relative_lift_pct,
# MAGIC     test_significance_status
# MAGIC FROM semantic_views.vw_fct_cro_funnel
# MAGIC ORDER BY test_variant, step_number;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 6 & 7: Operational SLA & A/B Experimentation Semantic Views
# MAGIC * **Objective:** Expose clean facts for infrastructure reliability and product-led growth experiments.
# MAGIC * **Key Actions:**
# MAGIC   * Defines `vw_fct_telemetry_sla` with automated SLA compliance classification based on tail latency ($p95$) and error percentage thresholds.
# MAGIC   * Formats `vw_fct_cro_funnel` with descriptive hypothesis test outcomes for clear non-technical interpretation.
# MAGIC * **Business & Architectural Value:** Provides Engineering VPs with instant operational SLA visibility while giving Product Managers verified, audit-ready A/B testing conclusions.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Semantic Layer Auditing**

# COMMAND ----------

views = [
    f"{catalog_name}.semantic_views.vw_dim_customers",
    f"{catalog_name}.semantic_views.vw_fct_mrr_waterfall",
    f"{catalog_name}.semantic_views.vw_fct_telemetry_sla",
    f"{catalog_name}.semantic_views.vw_fct_cro_funnel"
]

print(f"\n{'SEMANTIC VIEW NAME':<55} | {'RECORD COUNT':>15}")
print("-" * 73)
for v in views:
    cnt = spark.table(v).count()
    print(f"{v:<55} | {cnt:>15,}")
print("-" * 73)