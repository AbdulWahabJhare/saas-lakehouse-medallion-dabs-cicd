# Databricks notebook source
# MAGIC %md
# MAGIC # 04. Machine Learning: Churn Propensity Prediction Model
# MAGIC Trains a Logistic Regression model via PySpark MLlib and logs runs to MLflow.

# COMMAND ----------
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

# COMMAND ----------
# 1. Parameter Extraction & Catalog Configuration
dbutils.widgets.text("env", "dev", "Target Environment")
env = dbutils.widgets.get("env")

catalog_name = f"nexusmetrics_{env}"
gold_schema = "gold"

spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {gold_schema}")

# COMMAND ----------
# 2. Load Input Feature Data & Dynamically Detect Numeric Columns
df_features = spark.table(f"{catalog_name}.{gold_schema}.gold_customer_360_features")

print("Available columns in table:", df_features.columns)

# Exclude identifier, label, and timestamp columns from feature selection
excluded_cols = ["customer_id", "is_churned", "processed_at", "updated_at", "created_at", "inference_timestamp"]
numeric_types = ["IntegerType", "LongType", "DoubleType", "FloatType", "DecimalType", "ByteType", "ShortType"]

feature_cols = [
    field.name for field in df_features.schema.fields 
    if any(num_t in str(field.dataType) for num_t in numeric_types) and field.name not in excluded_cols
]

# Fallback: if data types are not strictly typed numeric, pick up to two non-excluded columns
if not feature_cols:
    feature_cols = [c for c in df_features.columns if c not in excluded_cols][:2]

print("Selected feature columns for model:", feature_cols)

# Ensure numeric cast for selected features
df_prep = df_features
for col_name in feature_cols:
    df_prep = df_prep.withColumn(col_name, F.col(col_name).cast("double"))

# Derive binary churn label if not present
if "is_churned" not in df_prep.columns:
    primary_feat = feature_cols[0]
    df_prep = df_prep.withColumn(
        "is_churned",
        F.when(F.col(primary_feat) <= 0, 1.0).otherwise(0.0)
    )
else:
    df_prep = df_prep.withColumn("is_churned", F.col("is_churned").cast("double"))

# Prepare modeling dataset and handle null values
df_dataset = df_prep.select(["customer_id", "is_churned"] + feature_cols).dropna()

# 80/20 Train-Test Split
train_df, test_df = df_dataset.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------
# 3. Assemble Machine Learning Pipeline
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="is_churned",
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.5
)

ml_pipeline = Pipeline(stages=[assembler, scaler, lr])

# COMMAND ----------
# 4. Train Model and Track Experiments in MLflow
mlflow.set_experiment(f"/Shared/nexusmetrics_churn_{env}")

with mlflow.start_run(run_name=f"churn_logistic_regression_{env}") as run:
    # Train pipeline
    model = ml_pipeline.fit(train_df)
    
    # Run inference on holdout test set
    test_predictions = model.transform(test_df)
    
    # Evaluate performance
    evaluator_roc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="is_churned",
        metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="is_churned",
        metricName="areaUnderPR"
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="is_churned",
        metricName="accuracy"
    )
    
    auc_roc = evaluator_roc.evaluate(test_predictions)
    auc_pr = evaluator_pr.evaluate(test_predictions)
    accuracy = evaluator_acc.evaluate(test_predictions)
    
    # Log hyperparameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("maxIter", 20)
    mlflow.log_param("regParam", 0.01)
    mlflow.log_param("elasticNetParam", 0.5)
    mlflow.log_param("feature_columns", feature_cols)
    
    # Log metrics
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model artifact
    mlflow.spark.log_model(model, "spark_model")
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

# COMMAND ----------
# 5. Score Full Dataset & Persist to Gold Delta Table
scored_df = model.transform(df_dataset)

# Extract probability for churned class (index 1)
extract_prob_udf = F.udf(lambda v: float(v[1]), DoubleType())

df_final_scored = (
    scored_df
    .withColumn("churn_risk_score", extract_prob_udf(F.col("probability")))
    .select(
        "customer_id",
        *feature_cols,
        "is_churned",
        "churn_risk_score",
        F.col("prediction").cast("integer").alias("predicted_churn_flag"),
        F.current_timestamp().alias("inference_timestamp")
    )
)

output_table = f"{catalog_name}.{gold_schema}.fct_churn_propensity_scored"
df_final_scored.write.mode("overwrite").format("delta").saveAsTable(output_table)

print(f"Successfully saved {df_final_scored.count()} predictions to {output_table}.")
