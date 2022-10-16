# The imports, above, allow us to access SparkML features specific to linear regression
from __future__ import print_function 
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, Word2Vec, IDF, Tokenizer, StopWordsRemover, CountVectorizer, Normalizer, HashingTF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.ml.feature import SQLTransformer
import pyspark
import sys

if len(sys.argv) != 2:
  raise Exception("Exactly 1 arguments are required: <inputUri> <outputUri>")

outputUri=sys.argv[1]

sc = SparkContext()
spark = SparkSession(sc)


filepath = "gs://big_data_dataset/yelp.json"
df = spark.read.json(filepath)

print((df.count(), len(df.columns)))

df.printSchema()

df.show()


# train, test = raw_data.randomSplit([0.8,0.2], 2020)
#train, test = stratified_split_train_test(df, frac=0.8, label="stars", join_on = 'user_id', seed=2020)

########################################################### Feature Engineering ########################################################################


## Building pipeline on Tfidf features
print("**********************************************************************************************")
print("Pipeline is Started")
print("**********************************************************************************************")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashtf = CountVectorizer(inputCol="filtered", outputCol="hash_tf")
idf = IDF(inputCol="hash_tf", outputCol="tfidf")
norm = Normalizer(inputCol = 'tfidf', outputCol = 'norm_tfidf')


tokenizer = Tokenizer(inputCol="text", outputCol="words")

token_df = tokenizer.transform(df)

stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

stop_df = stopwordsRemover.transform(token_df)

word2Vec = Word2Vec(vectorSize=300, minCount=5, inputCol="filtered", outputCol="wordvec")

norm_df = word2Vec.fit(stop_df).transform(stop_df)

clean_data = VectorAssembler(inputCols = ['wordvec', 'funny'], outputCol = 'features')

raw_dataset = clean_data.transform(norm_df)

dataset = raw_dataset.withColumn("label", raw_dataset["stars"]-1)

print((dataset.count(), len(dataset.columns)))

dataset.printSchema()

dataset.show()

(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed = 2020)
# (testData, aaData) =  validData.randomSplit([0.1, 0.9], seed = 100)




print("**********************************************************************************************")
print("Pipeline is done")
print("**********************************************************************************************")


print("**********************************************************************************************")
print("Training Started")
print("**********************************************************************************************")

nb = NaiveBayes(labelCol = 'label')
# lr = LogisticRegression(featuresCol="features", labelCol = 'label')
# lr = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=64)


model = nb.fit(trainingData)
# model = lr.fit(trainingData)

print("**********************************************************************************************")
print("Training Done")
print("**********************************************************************************************")

predictions_train = model.transform(trainingData)
predictions_test = model.transform(testData)

print("**********************************************************************************************")
print("Predictions on Test data is done, printing accuracy is remaining")
print("**********************************************************************************************")

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_train = evaluator.evaluate(predictions_train)
accuracy_test = evaluator.evaluate(predictions_test)


print("**********************************************************************************************")
print("Train accuracy without cross validation = %g" % (accuracy_train))
print("Test accuracy without cross validation = %g" % (accuracy_test))
print("**********************************************************************************************")

## Save Model
print("**********************************************************************************************")
print("Model is saving")
print("**********************************************************************************************")

# model.save(sys.argv[1] + '/Log_NBnorm_FullTrainedModel')

print("**********************************************************************************************")
print("Model Saved")
print("**********************************************************************************************")

