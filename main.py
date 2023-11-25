from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize a Spark session
spark = SparkSession.builder.appName(
    "CancerDiagnosisRandomForest").getOrCreate()

# Loading Dataset
# Replace 'path_to_dataset' with the actual path to your CSV file
data = spark.read.csv('./project3_data.csv', header=True, inferSchema=True)

# Preprocessing
# Convert categorical labels to numbers
labelIndexer = StringIndexer(
    inputCol="diagnosis", outputCol="indexedLabel").fit(data)

# Assemble feature vectors (assuming all other columns are features)
featureCols = data.columns
featureCols.remove('diagnosis')
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# Split the data into training and test sets
(trainData, testData) = data.randomSplit([0.7, 0.3])

# Training the Random Forest Model
# Define the Random Forest model
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, rf])

# Train model
model = pipeline.fit(trainData)

# Evaluating the Model
# Make predictions
predictions = model.transform(testData)

# Select prediction, true label, and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test Accuracy = %g" % (accuracy))
