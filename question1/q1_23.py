from pyspark.sql import SparkSession
import numpy as np
import time
spark = SparkSession.builder.master("local[20]").config("spark.local.dir","/fastdata/acq18lp").appName("COM6012 Decision Trees Regression").getOrCreate()
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

(trainingData, testData) = data.randomSplit([0.7, 0.3], 42)
trainingData = trainingData.cache()
testData = testData.cache()

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Binarizer


def findmodel(algo, bin, log):
  model = algo.fit(trainingData)
  predictions = model.transform(testData)
  if bin:
    binarizer = Binarizer(threshold = 0.5, inputCol="prediction_c", outputCol="prediction")
    predictions = binarizer.transform(predictions) 
  accuracy = evaluatorM.evaluate(predictions)
  auc = evaluatorB.evaluate(predictions)
  print("Accuracy:", accuracy)
  print("Area Under ROC:", auc)
  print("Top Features")
  if log:
    fi = model.coefficients.values
    for i in np.abs(fi).argsort()[-3:][::-1]:
      print(schemaNames[i+1], end=" ")
    print("")

  else:
    fi = model.featureImportances

    imp_feat = np.zeros(ncolumns-1)
    imp_feat[fi.indices] = fi.values
    for i in imp_feat.argsort()[-3:][::-1]:
        print(schemaNames[i+1], end=" ")
    print("")
  
  return model

evaluatorB = BinaryClassificationEvaluator(labelCol="labels", rawPredictionCol="prediction", metricName="areaUnderROC") 
evaluatorM= MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
binarizer = Binarizer(threshold = 0.5, inputCol="prediction_c", outputCol="prediction")
start = time.time()

print("Decision Tree Classifier")
dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features", maxDepth = 10, impurity = 'gini')
model = findmodel(dt,False,False)
stop = time.time()
print("Time:", (stop-start)//60, "minutes", (stop-start)%60, "seconds")
start = time.time()
print("Decision Tree Regressor")
dtr = DecisionTreeRegressor(labelCol="labels", featuresCol="features", predictionCol='prediction_c', maxDepth = 10, maxBins = 32)
model = findmodel(dtr,True,False)
stop = time.time()
print("Time:", (stop-start)//60, "minutes", (stop-start)%60, "seconds")
start = time.time()
print("Logistic Regression")
lr = LogisticRegression(featuresCol='features', labelCol='labels',predictionCol='prediction', family="binomial",maxIter = 20 , regParam = 0.0)
model = findmodel(lr,False,True)
stop = time.time()
print("Time:", (stop-start)//60, "minutes", (stop-start)%60, "seconds")

spark.stop()