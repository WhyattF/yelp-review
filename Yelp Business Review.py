#!/usr/bin/env python
# coding: utf-8


import findspark
findspark.init()

import re
import nltk
from pyspark import SparkConf, SparkContext
from nltk.corpus import stopwords
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF 
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import pyspark.sql as SQL
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('YelpBusinessReview').getOrCreate()


df = spark.read.format("csv").option("header", "true").option("multiline","true").load("yelp_business.csv") 
df.printSchema()
df.show()


business_df = df.select("business_id","review_count","stars","categories")
business_df.show()



li = ["5.0","4.5","4.0","3.5","3.0","2.5","2.0","1.5","1.0"]
business_review = business_df.filter(df.stars.isin(li))
business_review.show()


from pyspark.sql import functions as f
from pyspark.sql.functions import split,col,explode,count
from pyspark.sql.functions import length

#business_review_split = business_review.withColumn('categories_split', (f.split(f.col('categories'), ';')))
#business_review_split.show()

business_split = business_review.withColumn("category_split",split(col("categories"),";").getItem(0))
business_split.show()


pandasDF = business_split.toPandas()
pandasDF.head()

#here I am changing the types (reviews counts and stars to integers)
from pyspark.sql.types import IntegerType
business_split = business_split.withColumn("stars", business_split["stars"].cast(IntegerType()))


business_split = business_split.withColumn("review_count", business_split["review_count"].cast(IntegerType()))
business_split.printSchema()


#here I am doing a average star rating and sum of review count by categories
from pyspark.sql import functions as F
from pyspark.sql.functions import desc

business_agg = business_split.groupBy("category_split").agg(F.mean('stars'), F.sum('review_count')).sort(col("sum(review_count)").desc())
business_agg.show()


#sorting based on average star ratings - higher star ratings

business_split.groupBy("category_split").agg(F.mean('stars'), F.sum('review_count')).sort(col("avg(stars)").desc()).show()


#sorting based on average star ratings - lower star ratings
from pyspark.sql.functions import asc
business_split.groupBy("category_split").agg(F.mean('stars'), F.sum('review_count')).sort(col("avg(stars)").asc()).show()
#shows the middle ranked business based on review count
print('middle')
business_middle_rank = business_agg.filter((F.col('avg(stars)') > 3.0) & (f.col('avg(stars)') < 4.0))
business_middle_rank.show()
#shows the highest ranked businesses based on review count
print('high')
business_highest_rank = business_agg.filter((F.col('avg(stars)') > 4.0) & (f.col('avg(stars)') < 5.1))
business_highest_rank.show()
#shows the lowest ranked businesses based on review count
print('low')
business_lowest_rank = business_agg.filter((F.col('avg(stars)') > 0.0) & (f.col('avg(stars)') < 3.0))
business_lowest_rank.show()


pandasDF = business_agg.toPandas()
pandasDF.head()

l = pandasDF.nlargest(10, 'sum(review_count)')
f, ax = plt.subplots(figsize=(20, 5))
ax = sns.barplot(x = "category_split",y="avg(stars)",data = l, ci = None)
plt.title("Top Category Avg Rating", size=12)


l = pandasDF.nlargest(10, 'sum(review_count)')
f, ax = plt.subplots(figsize=(20, 5))
ax = sns.barplot(x = "category_split",y="sum(review_count)",data = l, ci = None)
plt.title("Top Category Review", size=12)

