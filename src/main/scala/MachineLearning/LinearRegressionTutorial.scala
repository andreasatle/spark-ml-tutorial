package MachineLearning

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors

object LinearRegressionTutorial extends App {

  // Create a new SparkSession
  val spark = SparkSession
    .builder()
    .appName("Logistic Regression Tutorial")
    .master("local[*]")
    .getOrCreate()

  // Use implicits
  import spark.implicits._

  // I cheat here, by naming the labels directly in the data.
  // I'm fitting a line to square data.
  // The exact least square solution to y = c0 + c1*x is
  // y = 6*x - 7. This has been validated in octave.
  // This makes no sense, but the solution is correct.
  val train = Seq(
    (1.0, 1.0),
    (2.0, 4.0),
    (3.0, 9.0),
    (4.0, 16.0),
    (5.0, 25.0)
  ).toDF("x", "label")

  // Read test data
  // I choose the data as far from the train data without
  // extrapolating.
  val testWithoutLabel = Seq(
    (1.5),
    (2.5),
    (3.5),
    (4.5)
  ).toDF("x")

  // Create an Array with features
  // This is the way most ML optimizers wants the data.
  val assembler = new VectorAssembler()
    .setInputCols(Array("x"))
    .setOutputCol("features")

  // Create a LinearRegression instance
  val linReg = new LinearRegression()

  // Prepare the data, and linear regression
  // The pipeline is in a weird state before fitting the train data.
  val pipeline = new Pipeline()
    .setStages(Array(assembler, linReg))

  // Fit the training data.
  val model = pipeline.fit(train)

  // Predict the transformed test data without label.
  // I suppose you need the label if you want to use
  // an evaluator at the end to check e.g. the accuracy.
  model.transform(testWithoutLabel).show
}
