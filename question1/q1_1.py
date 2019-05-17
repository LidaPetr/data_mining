from pyspark.sql import SparkSession
import numpy as np
spark = SparkSession.builder.master("local[10]").config("spark.local.dir","/fastdata/acq18lp").appName("COM6012 Decision Trees Regression").getOrCreate()
sc = spark.sparkContext

rawdata = spark.read.option('header', "false").csv('./Data/HIGGS.csv.gz')

schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata = rawdata.withColumn(schemaNames[i], rawdata[schemaNames[i]].cast(DoubleType()))
rawdata = rawdata.withColumnRenamed('_c0', 'labels')

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns], outputCol = 'features') 
raw_plus_vector = assembler.transform(rawdata)

data = raw_plus_vector.select('features','labels')

(configData, _) = data.randomSplit([0.25, 0.75], 42)

(trainingData, testData) = configData.randomSplit([0.7, 0.3], 42)
trainingData = trainingData.cache()
testData = testData.cache()

def metrics(pipeline, paramGrid, evaluator):

  crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator= evaluator,
                            numFolds=5) 
  
  return crossval


from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

evaluatorB = BinaryClassificationEvaluator(labelCol="labels", rawPredictionCol="prediction", metricName="areaUnderROC") 
evaluatorM= MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")



dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features")
pipeline = Pipeline(stages=[dt])

paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10]) \
    .addGrid(dt.impurity, ['gini', 'entropy'])\
    .build()
    
print("Decision Tree Classifier, Metric: Area Under ROC")
crossval = metrics(pipeline,paramGrid,evaluatorB)

cvModel = crossval.fit(trainingData)
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Area Under ROC:",np.max(cvModel.avgMetrics))


print("Decision Tree Classifier, Metric: Accuracy")

crossval = metrics(pipeline,paramGrid,evaluatorM)
cvModel = crossval.fit(trainingData)
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Accuracy:",np.max(cvModel.avgMetrics))

from pyspark.ml.feature import Binarizer

dtr = DecisionTreeRegressor(labelCol="labels", featuresCol="features", predictionCol='prediction_c')
binarizer = Binarizer(threshold = 0.5, inputCol="prediction_c", outputCol="prediction")
paramGrid = ParamGridBuilder() \
    .addGrid(dtr.maxDepth, [5, 10]) \
    .addGrid(dtr.maxBins, [16,32]) \
    .build()
pipeline = Pipeline(stages=[dtr, binarizer])

print("Decision Tree Regressor, Metric: Area Under ROC")
crossval = metrics(pipeline,paramGrid,evaluatorB)
cvModel = crossval.fit(trainingData)
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Area Under ROC:",np.max(cvModel.avgMetrics))

print("Decision Tree Regressor, Metric: Accuracy")
crossval = metrics(pipeline,paramGrid,evaluatorM)
cvModel = crossval.fit(trainingData)
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Accuracy:",np.max(cvModel.avgMetrics))

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol='features', labelCol='labels',predictionCol='prediction', family="binomial")
paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10, 20]) \
    .addGrid(lr.regParam, [0.0,0.5]) \
    .build()
    
pipeline = Pipeline(stages=[lr])

print("Logistic Regression, Metric: Area Under ROC")
crossval = metrics(pipeline,paramGrid,evaluatorB)
cvModel = crossval.fit(trainingData)
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Area Under ROC:",np.max(cvModel.avgMetrics))

print("Logistic Regression, Metric: Accuracy")
crossval = metrics(pipeline,paramGrid,evaluatorM)
cvModel = crossval.fit(trainingData)
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Accuracy:",np.max(cvModel.avgMetrics))

spark.stop()