package MachineLearning

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object IrisClassification extends App {
  // Create a SparkSession
  val spark = SparkSession
    .builder()
    .appName("Iris Logistic Regression Tutorial")
    .config("spark.master", "local[*]")
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

  // Split the dataset into training and test sets
  val Array(train, test) =
    iris.randomSplit(Array(0.8, 0.2), seed = 4711)

  // Create a logistic regression model
  val lr = new LogisticRegression()
    .setMaxIter(25)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  // Convert numerical labels back to categorical labels for evaluation
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  /*
    A pipeline in Spark ML consists of two types of stages:
      transformer and estimator.

    A transformer is a data processing stage that takes one or more
    input columns, applies a transformation function to them,
    and outputs one or more output columns.

    A transformer does not have any trainable parameters and can be
    used for tasks such as feature scaling, encoding categorical variables,
    and generating new features. Examples of transformers include
    `StringIndexer`, `OneHotEncoder`, and `VectorAssembler`.

    On the other hand, an estimator is a machine learning algorithm that
    can be trained on data to produce a model. Unlike a transformer,
    an estimator has trainable parameters that are learned from the input
    data during training. Examples of estimators include
    `LinearRegression`, `RandomForestClassifier`, and `KMeans`.

    The pipeline in Scala Spark ML can be created by chaining together
    multiple transformers and estimators using the `Pipeline` class.
    The `Pipeline` class takes a list of stages as input, where each stage
    is a transformer or an estimator.
   */

  // Fit the model to the training data
  val pipeline = new Pipeline()
    .setStages(Array(assembler, labelIndexer, lr, labelConverter))
  val model = pipeline.fit(train)

  // Make predictions on the test data
  val predictions = model.transform(test)
  predictions.show
  // Evaluate the model
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)
  println(s"Test set accuracy = $accuracy")

}
