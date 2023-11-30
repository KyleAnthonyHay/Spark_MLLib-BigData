from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("Random Forest Implementation\n\n")

spark = SparkSession.builder.appName(
    "CancerDiagnosisRandomForest").getOrCreate()

# Loading Dataset
data = spark.read.csv('./project3_data.csv', header=True, inferSchema=True)

# -------------- Preprocessing --------------
# Convert categorical labels to numbers
labelIndexer = StringIndexer(
    inputCol="diagnosis", outputCol="indexeddiagnosis").fit(data)

# Assemble feature vectors (assuming all other columns are features)
featureCols = data.columns
featureCols.remove('diagnosis')
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# Split the data into training and test sets
(trainData, testData) = data.randomSplit([0.7, 0.3])

# -------------- Training the Random Forest Model --------------
# Define the Random Forest model
rf = RandomForestClassifier(
    labelCol="indexeddiagnosis", featuresCol="features")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, rf])

# Train model
model = pipeline.fit(trainData)

# -------------- Make predictions --------------
predictions = model.transform(testData)

# Select prediction, true label, and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexeddiagnosis", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)


# -------------- Print Results --------------
print("Test Accuracy = %g" % (accuracy))
