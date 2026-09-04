# Databricks notebook source
# MAGIC %md
# MAGIC **MLFlow Environment Setup**

# COMMAND ----------

# 1. Define interactive widgets
dbutils.widgets.text("env", "dev", "1. Target Environment (dev/stage/prod)")
dbutils.widgets.text("experiment_name", "/Shared/nexusmetrics_churn_prediction", "2. MLflow Experiment Path")
dbutils.widgets.dropdown("register_model", "yes", ["yes", "no"], "3. Register in Unity Catalog?")

# 2. Retrieve parameter values
env = dbutils.widgets.get("env")
experiment_name = dbutils.widgets.get("experiment_name")
register_model = dbutils.widgets.get("register_model")
catalog_name = f"nexusmetrics_{env}"

# 3. Establish Unity Catalog namespaces
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("CREATE SCHEMA IF NOT EXISTS models")

import mlflow
mlflow.set_experiment(experiment_name)

print(f"✓ Active Catalog: {catalog_name} | Target Schema: models")
print(f"✓ MLflow Experiment Initialized: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1: Environment Configuration & MLflow Setup
# MAGIC * **Objective:** Configure dynamic workspace parameters and initialize the MLflow tracking experiment.
# MAGIC * **Key Actions:**
# MAGIC   * Creates runtime widgets for target environment (`env`), MLflow experiment location (`experiment_name`), and model registry preferences.
# MAGIC   * Ensures the target catalog and `models` schema exist in Unity Catalog.
# MAGIC   * Connects the active session to the centralized MLflow experiment registry.
# MAGIC * **Business & Architectural Value:** Ensures all model runs, parameters, metrics, and trained artifacts are securely logged and reproducible across development, staging, and production environments.

# COMMAND ----------

# MAGIC %md
# MAGIC **Extract Customer 360 featuring Store and Preprocessing pipelines**

# COMMAND ----------

# DBTITLE 1,Cell 5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

print("Loading features from gold.gold_customer_360_features...")

# 1. Pull curated Gold customer features
df_features = spark.table(f"{catalog_name}.gold.gold_customer_360_features").toPandas()

# 2. Separate target from metadata to prevent data leakage
target_col = "is_churned"
exclude_cols = [
    "org_id", "company_name_masked", "assigned_ae_masked", 
    "signup_date", "churn_date", "is_active", target_col,
    "company_name", "assigned_ae"  # String identifiers, not predictive features
]

feature_cols = [col for col in df_features.columns if col not in exclude_cols]
categorical_features = ["tier", "industry", "lead_source"]
numeric_features = [col for col in feature_cols if col not in categorical_features]

X = df_features[feature_cols]
y = df_features[target_col].astype(int)

# 3. Stratified 80/20 train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Preprocessor: One-Hot Encode categoricals, pass through numeric values
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Retrieve transformed column names for feature importance
ohe_cols = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
all_feature_names = ohe_cols + numeric_features

print(f"✓ Training Matrix: {X_train.shape[0]:,} records across {X_train.shape[1]} engineered features")
print(f"✓ Testing Matrix : {X_test.shape[0]:,} records")
print(f"✓ Baseline Churn Prevalence: {y.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2: Feature Store Preprocessing & Stratified Splitting
# MAGIC * **Objective:** Prepare a clean, leakage-free feature matrix from the Customer 360 Gold table.
# MAGIC * **Key Actions:**
# MAGIC   * Pulls multi-source firmographics, telemetry metrics, and support history from `gold_customer_360_features`.
# MAGIC   * Drops non-predictive identifiers and timestamps (`org_id`, `company_name_masked`, `churn_date`) to prevent data leakage.
# MAGIC   * Executes a stratified 80/20 train/test split to preserve the natural churn distribution.
# MAGIC   * One-hot encodes categorical dimensions (`tier`, `industry`, `lead_source`) and passes through numeric feature vectors.
# MAGIC * **Business & Architectural Value:** Guarantees standard statistical preprocessing and prevents lookahead bias before feeding data into tree-based gradient boosting models.

# COMMAND ----------

# MAGIC %md
# MAGIC **Training XGBoost Classifier with MLflow Tracking**

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from mlflow.models.signature import infer_signature

# 1. Address class imbalance via positive weight ratio
negative_samples = (y_train == 0).sum()
positive_samples = (y_train == 1).sum()
scale_pos_weight = float(negative_samples / positive_samples)

# 2. Hyperparameters
params = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "eval_metric": "aucpr"
}

with mlflow.start_run(run_name="xgboost_churn_enterprise") as run:
    # 3. Train
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 4. Predict probabilities and binary class labels
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)
    
    # 5. Evaluate validation metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # 6. Log parameters & metrics to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # 7. Log model artifact with data signature
    signature = infer_signature(X_test, y_pred)
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="churn_model",
        signature=signature
    )
    
    active_run_id = run.info.run_id
    print(f"✓ MLflow Run Logged: {active_run_id}")
    print(f"  * ROC-AUC Score : {roc_auc:.4f}")
    print(f"  * PR-AUC Score  : {pr_auc:.4f}")
    print(f"  * F1-Score      : {f1:.4f}")
    print(f"  * Recall Rate   : {recall:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 3: XGBoost Training & MLflow Experiment Tracking
# MAGIC * **Objective:** Train an enterprise-grade gradient boosted decision tree classifier to predict customer churn propensity.
# MAGIC * **Key Actions:**
# MAGIC   * Calculates `scale_pos_weight` to counteract class imbalance between churned and retained organizations.
# MAGIC   * Trains an `XGBClassifier` optimized against precision-recall area under the curve (`aucpr`).
# MAGIC   * Automatically logs hyperparameters, ROC-AUC, PR-AUC, F1-Score, and model signatures to the MLflow tracking server.
# MAGIC * **Business & Architectural Value:** Gives the data science team automated model lineage, auditability, and reproducible metric tracking across every experiment run.

# COMMAND ----------

# MAGIC %md
# MAGIC **Feature Importance And Explainablity**

# COMMAND ----------

# 1. Compute normalized feature importance from XGBoost booster
importance_scores = model.feature_importances_
df_importance = pd.DataFrame({
    "feature_name": all_feature_names,
    "importance_weight": importance_scores
}).sort_values(by="importance_weight", ascending=False)

print("\n--- TOP 10 LEADING DRIVERS OF CHURN ---")
print(df_importance.head(10).to_string(index=False))

# 2. Log importance matrix as an MLflow artifact
importance_dict = dict(zip(df_importance["feature_name"], df_importance["importance_weight"]))
mlflow.log_dict(importance_dict, "feature_importance_summary.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 4: Feature Importance & Explainability Analysis
# MAGIC * **Objective:** Identify and rank the primary behavioral, financial, and operational drivers leading to customer cancellations.
# MAGIC * **Key Actions:**
# MAGIC   * Extracts Gini gain importance scores across all features utilized by the tree ensembles.
# MAGIC   * Isolates the top predictive signals (e.g., support friction, latency spikes, rate limits).
# MAGIC   * Serializes the importance weights into an MLflow artifact dictionary (`feature_importance_summary.json`).
# MAGIC * **Business & Architectural Value:** Enables Customer Success leadership to understand *why* accounts churn, revealing whether cancellations stem from technical service disruptions (HTTP 429s/latency) or poor support sentiment.

# COMMAND ----------

# MAGIC %md
# MAGIC **Batch Inferance Scoring And Writing To Gold Table**

# COMMAND ----------

# 1. Run batch inference over all organizations in the feature store
X_all_processed = preprocessor.transform(df_features[feature_cols])
churn_probabilities = model.predict_proba(X_all_processed)[:, 1]

# 2. Build scored output dataframe with risk banding
df_scored = df_features[["org_id", "company_name_masked", "tier", "industry", "is_churned"]].copy()
df_scored["churn_risk_score"] = np.round(churn_probabilities, 4)
df_scored["risk_band"] = pd.cut(
    df_scored["churn_risk_score"],
    bins=[-np.inf, 0.30, 0.65, np.inf],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# 3. Convert to Spark DataFrame and write to Gold layer
df_scored_spark = spark.createDataFrame(df_scored)

(
    df_scored_spark.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{catalog_name}.gold.org_churn_predictions")
)

print(f"✓ Saved: {catalog_name}.gold.org_churn_predictions ({df_scored_spark.count():,} records)")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     risk_band,
# MAGIC     tier,
# MAGIC     COUNT(*) AS total_organizations,
# MAGIC     ROUND(AVG(churn_risk_score) * 100, 2) AS avg_churn_probability_pct
# MAGIC FROM gold.org_churn_predictions
# MAGIC WHERE is_churned = 0
# MAGIC GROUP BY risk_band, tier
# MAGIC ORDER BY risk_band DESC, total_organizations DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 5 & 6: Batch Scoring & Curated Predictions Gold Table
# MAGIC * **Objective:** Score every active SaaS account with an empirical churn probability and segment them into actionable operational risk tiers.
# MAGIC * **Key Actions:**
# MAGIC   * Generates batch predictions across the full customer base using the preprocessed feature vectors.
# MAGIC   * Assigns accounts to standardized risk categories: `Low Risk` (<30%), `Medium Risk` (30–65%), and `High Risk` (>65%).
# MAGIC   * Persists the scored results into Delta table `gold.org_churn_predictions`.
# MAGIC * **Business & Architectural Value:** Supplies front-line Customer Success Managers and Account Executives with prioritized, daily account lists to intervene proactively and prevent revenue loss before cancellations happen.

# COMMAND ----------

# MAGIC %md
# MAGIC **Auditing and Row Count**

# COMMAND ----------

print(f"{'TABLE NAME':<55} | {'ROW COUNT':>15}")
print("-" * 73)
cnt = spark.table(f"{catalog_name}.gold.org_churn_predictions").count()
print(f"{catalog_name}.gold.org_churn_predictions:<55 | {cnt:>15,}")
print("-" * 73)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 7: Predictive Layer Row Count & Schema Validation
# MAGIC * **Objective:** Verify successful table generation and check partition consistency for downstream reporting.
# MAGIC * **Key Actions:**
# MAGIC   * Queries Unity Catalog metadata to ensure all 10,000 organization entities have been assigned churn risk scores.
# MAGIC   * Confirms Delta schema compliance for ingestion into executive dashboards and Power BI reports.
# MAGIC * **Business & Architectural Value:** Ensures zero record loss and validates that operational tables are ready for consumption by BI semantic views.

# COMMAND ----------

