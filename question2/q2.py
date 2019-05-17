import os
import time

from pyspark.sql import SparkSession
import numpy as np
spark = SparkSession.builder.master("local[20]").config("spark.local.dir","/fastdata/acq18lp").appName("COM6012 Decision Trees Regression").getOrCreate()

sc = spark.sparkContext

start = time.time()

rawdata = spark.read.option('header', "true").csv('/data/acq18lp/Data/train_set.csv')

from pyspark.sql.functions import when, lit, col

schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)

def replace(column, value):
    return when(column != value, column).otherwise(lit(None))

for i in range(ncolumns):
    rawdata = rawdata.withColumn(schemaNames[i], replace(col(schemaNames[i]), "?"))

print("Remove Null values")
rawdata = rawdata.na.drop()

# Change schema types to integers or doubles
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
ints = [0,1,2,3,4,20]
doubles = [21,22,23,24,25,26,27,28,30,31,32,33,34]
strings = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,29]
for i in ints:
    rawdata = rawdata.withColumn(schemaNames[i], rawdata[schemaNames[i]].cast(IntegerType()))
for i in doubles:
    rawdata = rawdata.withColumn(schemaNames[i], rawdata[schemaNames[i]].cast(DoubleType()))

# save the name of the columns with strings
string_list = []
for i in strings:
    string_list.append(schemaNames[i])

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# replace the strings with stringIndexer

print("Replace Categorical Features")

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in string_list]
pipeline = Pipeline(stages=indexers)
indexed = pipeline.fit(rawdata).transform(rawdata)

string_list.append("Claim_Amount")

final_list = []
for i in indexed.schema.names:
    if i not in string_list:
        final_list.append(i)
        
#keep the Claim_Amount column as the last column
final_list.append("Claim_Amount")   

# remove the columns with strings
rawdata = indexed.select([column for column in indexed.columns if column in final_list])


from pyspark.sql.functions import lit
from pyspark.ml.feature import VectorAssembler

print("Make the dataset balanced")
# balance the dataset by removing instances with Claim Amount equal to zero
zero_data = rawdata.filter(rawdata['Claim_Amount'] == 0.0)
nzero_data = rawdata.filter(rawdata['Claim_Amount'] > 0.0)
zero_data_c = zero_data.count()
nzero_data_c = nzero_data.count()
zero_data = zero_data.sample(False, nzero_data_c/zero_data_c, 42)
sampled_data = zero_data.union(nzero_data)

assembler = VectorAssembler(inputCols = final_list[0:ncolumns-1], outputCol = 'features') 
raw_plus_vector = assembler.transform(sampled_data)

data = raw_plus_vector.select('features','Claim_Amount')



(traindata, testdata) = data.randomSplit([0.7, 0.3], 42)


# First predictive model
print("Linear Regression")

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='Claim_Amount', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(traindata)

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
traindata.describe().show()

lr_predictions = lr_model.transform(testdata)
test_result = lr_model.evaluate(testdata)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)




from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import Binarizer

# Second predictive model
print("Decision Tree Classifier")
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", predictionCol='prediction_c', maxBins=800)
binarizer = Binarizer(threshold = 0.0001, inputCol='Claim_Amount', outputCol='label')

pipeline = Pipeline(stages=[binarizer,dt])
dtModel = pipeline.fit(traindata)
# Make predictions on test data using the Transformer.transform() method.
predictions = dtModel.transform(testdata) 

non_zero_train = traindata.filter(traindata['Claim_Amount'] >0.0)
non_zero_test = predictions.filter(predictions['prediction_c'] >0.0)

print("Generalized Linear Regression with gamma family")

from pyspark.ml.regression import GeneralizedLinearRegression
glm_gamma = GeneralizedLinearRegression(featuresCol='features', labelCol='Claim_Amount', maxIter=50,\
                                          family='gamma', link='log')
glm_model = glm_gamma.fit(non_zero_train)
predictions = glm_model.transform(non_zero_test)

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator\
      (labelCol="Claim_Amount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)

end = time.time()