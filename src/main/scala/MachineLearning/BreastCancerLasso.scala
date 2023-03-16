package MachineLearning
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression

import org.apache.spark.sql.types.{
  StructType,
  StructField,
  IntegerType,
  LongType,
  DoubleType,
  StringType
}
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.evaluation.RegressionEvaluator

object BreastCancerLasso extends App {
  // Create a SparkSession
  val spark = SparkSession
    .builder()
    .appName("Breast Cancer Lasso Tuturial")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  val breastCancerSchema = StructType(
    Array(
      StructField("id", LongType),
      StructField("clump_thickness", DoubleType),
      StructField("uniformity_of_cell_size", DoubleType),
      StructField("uniformity_of_cell_shape", DoubleType),
      StructField("marginal_adhesion", DoubleType),
      StructField("single_epithelial_cell_size", DoubleType),
      StructField("bare_nuclei", DoubleType),
      StructField("bland_chromatin", DoubleType),
      StructField("normal_nucleoli", DoubleType),
      StructField("mitsoses", DoubleType),
      StructField("class", StringType)
    )
  )
  // Read in breast cancer data and drop rows with missing values
  val data = spark.read
    .format("csv")
    .option("header", false)
    .option("nullValue", "?")
    .schema(breastCancerSchema)
    .load("data/breast-cancer-wisconsin.data")
    .na
    .drop()

  val assembler = new VectorAssembler()
    .setInputCols(
      Array(
        "clump_thickness",
        "uniformity_of_cell_size",
        "uniformity_of_cell_shape",
        "marginal_adhesion",
        "single_epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitsoses"
      )
    )
    .setOutputCol("features")
  val labelIndexer = new StringIndexer()
    .setInputCol("class")
    .setOutputCol("label")
    .fit(data)

  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Create a logistic regression model
  val lr = new LogisticRegression()
    .setMaxIter(250)
    .setRegParam(0.3)
    .setElasticNetParam(0.5)

  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, assembler, lr, labelConverter))

  // Split the dataset into training and test sets
  val Array(train, test) =
    data.randomSplit(Array(0.8, 0.2), seed = 4711)

  val model = pipeline.fit(train)

  // Make predictions on the test data
  val predictions = model.transform(test)
  predictions.show

  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

  val rmse = evaluator.evaluate(predictions)
  println(s"RMSE: $rmse")

}
