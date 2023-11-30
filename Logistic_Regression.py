from pyspark.ml.classification import LogisticRegression

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("Logistic Regression Implementation\n\n")

spark = SparkSession.builder.appName(
    "CancerDiagnosisRandomForest").getOrCreate()

# Loading Dataset
data = spark.read.csv('./project3_data.csv', header=True, inferSchema=True)

# Preprocessing
labelIndexer = StringIndexer(
    inputCol="diagnosis", outputCol="indexedDiagnosis").fit(data)

featureCols = data.columns
featureCols.remove('diagnosis')
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

(trainData, testData) = data.randomSplit([0.7, 0.3])

# Define and Train the Logistic Regression Model
lr = LogisticRegression(labelCol="indexedDiagnosis", featuresCol="features")

# Chain indexers and logistic regression in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, lr])

# Train the model
lrModel = pipeline.fit(trainData)

# Evaluate the Model
# Make predictions
predictions = lrModel.transform(testData)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedDiagnosis", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test Accuracy = %g" % (accuracy))
