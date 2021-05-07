#!/usr/bin/env python
# coding: utf-8




import re
import nltk
from pyspark import SparkConf, SparkContext
from nltk.corpus import stopwords
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF 
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import pandas
import matplotlib.pyplot as plt





spark = SparkSession.builder.appName('Yelp Review').getOrCreate()




df = spark.read.format("csv").option("header", "true").option("multiline","true").load("yelp_review.csv") 
df.printSchema()
df.show()




yelp_df = df.select("review_id","text","stars")
yelp_df.show()



yelp_df_clean = yelp_df.na.drop()
yelp_df_clean.show()
yelp_df_clean.printSchema()




#Here I am creating a list to only include star ratings of 0-5. 
li = ["5","4","3","2","1"]
yelp_review = yelp_df_clean.filter(df.stars.isin(li))
yelp_review.show()



#here I am removing all puncuations so it can make it easier to count words within each of the stars
from pyspark.sql.functions import regexp_replace, trim, col, lower

def removePunctuation(column):
    return lower(trim(regexp_replace(column,'\\p{Punct}',''))).alias('text')

#col:Returns a Column based on the given column name.
yelpDF = yelp_review.select('review_id', removePunctuation(col('text')),'stars')
yelpDF.show()




from pyspark.sql.functions import col, explode, regexp_replace, split, rank
cleaned_review_rating = yelpDF.withColumn("text",explode(split(regexp_replace(col("text"), "(.^\[)|(\]$)", " "), " ")))
cleaned_review_rating.show()




#removing all stopwords from text field 
stopword_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
yelpDF_clean = cleaned_review_rating.filter(~col('text').isin(stopword_list))
yelpDF_clean.show()




from pyspark.sql import functions as f
from pyspark.sql.functions import split,col,explode,count
from pyspark.sql.functions import length
from pyspark.sql.window import Window

final_yelpDF = yelpDF_clean.groupBy('text','stars').count().sort('stars','count', ascending=False)
final_yelpDF = final_yelpDF.na.drop()
final_yelpDF.show()




bottom = ["1","2"]
bottom_yelp_review = final_yelpDF.filter(df.stars.isin(bottom))
bottom_yelp_review.show()




top = ["5","4"]
top_yelp_review = final_yelpDF.filter(df.stars.isin(top))
top_yelp_review.show()



#used to push reviews to Pandas for word cloud analysis
#bottom_yelp_review.toPandas().to_csv('bottom_review.csv')



#used to push reviews to Pandas for word cloud analysis
#top_yelp_review.toPandas().to_csv('top_review.csv')




#exploratory analysis. I used this to find out how clean the data is. Therefore I used a filter to only investiate
#cases where the stars were actually 1-5. I will next change the datatype to integer so I can use in future plots 
yelp_stars_count = yelp_review.groupby("stars").count().sort(col("count").desc())
yelp_stars_count.show()

