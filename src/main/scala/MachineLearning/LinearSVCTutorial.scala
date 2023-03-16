package MachineLearning

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}

object LinearSVCTutorial extends App {
  // Create a new SparkSession
  val spark = SparkSession
    .builder()
    .appName("Linear SVC Tutorial")
    .master("local[*]")
    .getOrCreate()

  // Read iris dataset (took it from kaggle)
  val iris = spark.read
    .format("csv")
    .option("header", true)
    .option("inferSchema", true)
    .load("data/iris.csv")

  /*
    `StringIndexer` is a feature transformer in Scala Spark ML library
    that can convert a column of categorical strings into a column of
    numerical indices. It assigns a unique index to each unique value
    in the input column, starting from 0. The indices are ordered by
    frequency, so the most frequent value gets index 0. It can be useful
    for algorithms that work better with numerical input data, such as
    decision trees or logistic regression.
   */

  // Convert the categorical labels to numerical values
  val labelIndexer = new StringIndexer()
    .setInputCol("Species")
    .setOutputCol("label")
    .fit(iris)

  /*
    `VectorAssembler` is a feature transformer in Scala Spark ML library
    that can combine multiple columns of data into a single vector column.
    It takes a list of input column names and creates a new vector column
    where each element of the vector corresponds to one of the input columns.
    It can be useful for preparing data for algorithms that require a single
    input vector, such as linear regression or support vector machines.
   */

  // Create a feature vector by combining the four numerical features
  val assembler = new VectorAssembler()
    .setInputCols(
      Array("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")
    )
    .setOutputCol("features")

  val Array(train, test) = labelIndexer
    .transform(assembler.transform(iris))
    .select("features", "label")
    .randomSplit(Array(0.7, 0.3), seed = 1234)

  // Setup the SVM model. This is only for binary classification.
  val svm = new LinearSVC()
    .setMaxIter(25)
    .setRegParam(0.1)
    .setLabelCol("label")

  // instantiate the One Vs Rest Classifier.
  // Hm, the results are not very good, maybe OVR is not a good idea?!
  val ovr = new OneVsRest().setClassifier(svm)
  val model = ovr.fit(train)

  val predictionConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predicted_Species")
    .setLabels(labelIndexer.labels)

  val labelConverter = new IndexToString()
    .setInputCol("label")
    .setOutputCol("labeled_Species")
    .setLabels(labelIndexer.labels)

  // Evaluate the model on the testing data
  val predictions = labelConverter.transform(
    predictionConverter.transform(model.transform(test))
  )
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Accuracy: ${accuracy}")

  predictions.show(numRows = 100)

}
