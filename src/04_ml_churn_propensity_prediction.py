# Databricks notebook source
# MAGIC %md
# MAGIC # 04. Machine Learning: Churn Propensity Prediction Model
# MAGIC Trains a Logistic Regression model via PySpark MLlib and logs runs to MLflow.

# COMMAND ----------
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pyspark.sql.functions as F

# COMMAND ----------
# 1. Parameter Extraction
dbutils.widgets.text("env", "dev", "Target Environment")
env = dbutils.widgets.get("env")

catalog_name = f"nexusmetrics_{env}"
gold_schema = "gold"

spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {gold_schema}")

# COMMAND ----------
# 2. Load Input Feature Data from Gold Aggregations
df_features = spark.table(f"{catalog_name}.{gold_schema}.mrr_aggregations")

# Create a binary churn label for training demonstration if missing
# In production, this label comes from historical subscription cancellations
if "is_churned" not in df_features.columns:
    df_features = df_features.withColumn(
        "is_churned",
        F.when(F.col("mrr_amount") <= 0, 1.0)
         .when(F.col("active_subscriptions") == 0, 1.0)
         .otherwise(0.0)
    )

# Filter out null records for modeling stability
feature_cols = ["mrr_amount", "active_subscriptions"]
df_dataset = df_features.select(["customer_id", "is_churned"] + feature_cols).dropna()

# Split into 80% Training and 20% Testing sets
train_df, test_df = df_dataset.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------
# 3. Assemble Machine Learning Pipeline
# Step A: Assemble numerical features into a feature vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="skip"
)

# Step B: Scale features to zero-mean and unit variance
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Step C: Logistic Regression Classifier
lr = LogisticRegression(
    featuresCol="features",
    labelCol="is_churned",
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.5
)

# Combine into an end-to-end Pipeline
ml_pipeline = Pipeline(stages=[assembler, scaler, lr])

# COMMAND ----------
# 4. Train Model and Log to MLflow
experiment_path = f"/Users/{spark.conf.get('spark.databricks.workspaceUrl', 'default')}/churn_prediction_experiment"
mlflow.set_experiment(f"/Shared/nexusmetrics_churn_{env}")

with mlflow.start_run(run_name=f"churn_logistic_regression_{env}") as run:
    # Train the pipeline model
    model = ml_pipeline.fit(train_df)
    
    # Run predictions on the test holdout set
    predictions = model.transform(test_df)
    
    # Evaluate Model Metrics
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
    
    auc_roc = evaluator_roc.evaluate(predictions)
    auc_pr = evaluator_pr.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)
    
    # Log Hyperparameters to MLflow
    mlflow.log_param("maxIter", 20)
    mlflow.log_param("regParam", 0.01)
    mlflow.log_param("elasticNetParam", 0.5)
    mlflow.log_param("features", feature_cols)
    
    # Log Evaluation Metrics to MLflow
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log Model Artifact
    mlflow.spark.log_model(model, "spark_model")
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

# COMMAND ----------
# 5. Score Full Dataset & Write to Gold Output Table
# Extract probability vector's positive class (churn risk probability)
scored_df = model.transform(df_dataset)

# PySpark helper to pull the 2nd index from the probability vector (churn score between 0.0 and 1.0)
extract_prob_udf = F.udf(lambda v: float(v[1]), "double")

df_final_scored = (
    scored_df
    .withColumn("churn_risk_score", extract_prob_udf(F.col("probability")))
    .select(
        "customer_id",
        "mrr_amount",
        "active_subscriptions",
        "is_churned",
        "churn_risk_score",
        F.col("prediction").alias("predicted_churn_flag"),
        F.current_timestamp().alias("inference_timestamp")
    )
)

# Write output Delta table
output_table = f"{catalog_name}.{gold_schema}.fct_churn_propensity_scored"
df_final_scored.write.mode("overwrite").format("delta").saveAsTable(output_table)

print(f"Successfully scored {df_final_scored.count()} records into {output_table}.")
