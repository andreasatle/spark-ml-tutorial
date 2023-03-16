package MachineLearning

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

object LogisticRegressionTutorial extends App {
  // Create a SparkSession
  val spark = SparkSession
    .builder()
    .appName("Logistic Regression Tutorial")
    .config("spark.master", "local[*]")
    .getOrCreate()

  // Load training data on the libsvm format
  val data = spark.read
    .format("libsvm")
    .load("data/sample_binary_classification_data.txt")

  val Array(trainData, testData) =
    data.randomSplit(Array(0.75, 0.25), seed = 42)
  // Create an new LogisticRegression
  val lr = new LogisticRegression()
    .setMaxIter(15)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  // Create an evaluator to check the result on test data
  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

  // Fit the model
  val lrModel = lr.fit(trainData)
  val lrPredictions = lrModel.transform(testData)
  val lrRmse = evaluator.evaluate(lrPredictions)
  println(s"RMSE: $lrRmse")

  // Plot the training loss histogram
  val losses = lrModel.summary.objectiveHistory

  spark
    .createDataFrame(losses.zipWithIndex)
    .toDF("Training Loss", "Iteration")
    .show
}
