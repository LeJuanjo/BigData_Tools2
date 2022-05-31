# Databricks notebook source
# DBTITLE 1,Group Assignment 
# Group 3
# Members: GOPALAKRISHNAN Harikrishnan, MEDINAMARTINEZ Juanjose, NGUYEN Nguyet Han
# Big data tools 2021

# COMMAND ----------

pip install category_encoders

# COMMAND ----------

#initializing packages and functions
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, DateType
from pyspark.sql.functions import datediff, to_date, lit

# COMMAND ----------

business_path = "/FileStore/tables/parsed_business.json"
checkin_path = "/FileStore/tables/parsed_checkin.json"
review_path = "/FileStore/tables/parsed_review.json"
covid_path = "/FileStore/tables/parsed_covid.json"
tip_path = "/FileStore/tables/parsed_tip.json"
user_path = "/FileStore/tables/parsed_user.json"

# COMMAND ----------

#legacy format for time
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

# COMMAND ----------

# DBTITLE 1,Table: users
#Read in tip data
tip = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(user_path)

# COMMAND ----------

#set new types
user = user.withColumn("average_stars",user.average_stars.cast('float'))
user = user.withColumn("compliment_cool",user.compliment_cool.cast('short'))
user = user.withColumn("compliment_cute",user.compliment_cute.cast('short'))
user = user.withColumn("compliment_funny",user.compliment_funny.cast('short'))
user = user.withColumn("compliment_hot",user.compliment_hot.cast('short'))
user = user.withColumn("compliment_list",user.compliment_list.cast('short'))
user = user.withColumn("compliment_more",user.compliment_more.cast('short'))
user = user.withColumn("compliment_note",user.compliment_note.cast('short'))
user = user.withColumn("compliment_photos",user.compliment_photos.cast('short'))
user = user.withColumn("compliment_plain",user.compliment_plain.cast('short'))
user = user.withColumn("compliment_profile",user.compliment_profile.cast('short'))
user = user.withColumn("compliment_writer",user.compliment_writer.cast('short'))
user = user.withColumn("cool",user.cool.cast('short'))
user = user.withColumn("fans",user.fans.cast('short'))
user = user.withColumn("funny",user.funny.cast('short'))
user = user.withColumn("review_count",user.review_count.cast('short'))
user = user.withColumn("useful",user.useful.cast('short'))
user = user.withColumn("yelping_since",user.yelping_since.cast('date'))

# COMMAND ----------

from pyspark.sql.functions import mean, min, max, sum, datediff, to_date,current_date, col
user = user.withColumn("current_date",current_date())
user=user.withColumn("relationship_length",datediff(col("current_date"),col("yelping_since")))


# COMMAND ----------

# DBTITLE 1,Table: tip
#Read in tip data
tip = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(tip_path)

# COMMAND ----------

tip.show(3)

# COMMAND ----------

#extract the date and get the last date when a tip/compliment was made
tip = tip.withColumn('date', split(tip['date'], ' ').getItem(0))
tip = tip.withColumn("date", tip['date'].cast("Date"))
tip_df = tip.groupBy("business_id").agg(sum("compliment_count").alias("compliment_count"),\
                                    max("date").alias("max_date"))
tip_df.display()

# COMMAND ----------

#recency of the last comliment
tip_df = tip_df.withColumn("days_since last_tip",datediff(to_date(lit("2020-03-31")), tip_df.max_date))   
tip_df = tip_df.drop(col("max_date"))
tip_df.display()

# COMMAND ----------

# DBTITLE 1,Table: checkin
#Read in checkin data
checkin = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(checkin_path)

# COMMAND ----------

checkin.show(3)
###############
checkin.printSchema()

# COMMAND ----------

# Number of check-in per business
checkin1 = checkin.groupBy("business_id").agg(count("date").alias("nbr_checkin"))
checkin1.show(5)

# COMMAND ----------

# Calculate the last checkin date
checkin = checkin.withColumn('date', split(checkin['date'], ' ').getItem(0))
checkin = checkin.withColumn("date", checkin['date'].cast("Date"))
checkin_df = checkin.groupBy("business_id").agg(max("date").alias("max_date"))
checkin_df.display()

# COMMAND ----------

# Calculate date since last check and merge with checkin1
checkin_df = checkin_df.withColumn("days_since last_check",datediff(to_date(lit("2020-03-31")), checkin_df.max_date))   
checkin_df = checkin_df.drop(col("max_date"))
checkin_df = checkin_df.join(checkin1, "business_id", "left")
checkin_df.show(3)

# COMMAND ----------

# DBTITLE 1,Table: business
# Read in business data
business = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(business_path)

# COMMAND ----------

#Check the DF's schema
business.printSchema()
######################
business.display()

# COMMAND ----------

# Replace dots with dash to be able to select the columns
new_cols=(column.replace('.', '_') for column in business.columns)
business = business.toDF(*new_cols)

# COMMAND ----------

# Replace null with 0
business = business.withColumn("attributes_RestaurantsTakeOut",when((business.attributes_RestaurantsTakeOut == "False")|(business.attributes_RestaurantsTakeOut.isNull()),lit(0)).otherwise(lit(1)))
business = business.withColumn("attributes_RestaurantsDelivery",when((business.attributes_RestaurantsDelivery == "False")|(business.attributes_RestaurantsTakeOut.isNull()),lit(0)).otherwise(lit(1)))
business = business.withColumn("pseudo_target",when((business.attributes_RestaurantsTakeOut==1)|(business.attributes_RestaurantsDelivery ==1),lit(1)).otherwise(lit(0)))

# COMMAND ----------

# Split categories column into 5 distinct variable
business = business.withColumn('cat1', split(business['categories'], ',').getItem(0)) \
       .withColumn('cat2', split(business['categories'], ',').getItem(1)) \
       .withColumn('cat3', split(business['categories'], ',').getItem(2)) \
       .withColumn('cat4', split(business['categories'], ',').getItem(3)) \
       .withColumn('cat5', split(business['categories'], ',').getItem(4)) 

# COMMAND ----------

# Get count of both null and missing values 
missing_business = business.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in business.columns])
missing_business.display()

# COMMAND ----------

# Get Size and shape of the dataframe 
print((business.count(), len(business.columns)))

# COMMAND ----------

# Get the unique categories
category = business.select('categories')
individual_category = category.select(explode(split('categories', ',')).alias('category'))
individual_category.show(50)

# COMMAND ----------

#get prequency of the categories
individual_category = individual_category.groupby("category").agg(count("category").alias("frecuency")).sort(col("frecuency").desc()).limit(100)
individual_category.display()

# COMMAND ----------

#fix categories since some of them have blank spaces
individual_category = individual_category.withColumn('category', trim(individual_category.category))
individual_category.show(truncate =False)

# COMMAND ----------

#create a list of tags that no not belongto the leisure industry

exclude_list = ["Home Services","Automotive", "Event Planning & Services", "Fashion", "Home Services", "Professional Services", "Health & Medical", "Auto Repair", "Doctors", "Real Estate", "Local Services", "Women's Clothing", "Grocery" ,"Education", "Dentists", "Contractors", "Financial Services", "General Dentistry", "Pet Services", "Home Decor", "Furniture Stores", "Flowers & Gifts", "Tires", "Oil Change Stations", "Oil Change Stations"]
exclude_list

# COMMAND ----------

#create category list to filter base on them
catagory_50 =individual_category.select("category").collect()

catagory_50=[i[0] for i in catagory_50]

catagory_50

# COMMAND ----------

#exclude business that are not related to the industry
catagory_50.remove("Automotive")
catagory_50.remove("Home Services")
catagory_50.remove("Automotive")
catagory_50.remove("Event Planning & Services")
catagory_50.remove("Fashion")
catagory_50.remove("Home Services")
catagory_50.remove("Professional Services")
catagory_50.remove("Health & Medical")
catagory_50.remove("Auto Repair")
catagory_50.remove("Doctors")
catagory_50.remove("Real Estate")
catagory_50.remove("Local Services")
catagory_50.remove("Women's Clothing")
catagory_50.remove("Grocery")
catagory_50.remove("Education")
catagory_50.remove("Dentists")
catagory_50.remove("Contractors")
catagory_50.remove("Financial Services")
catagory_50.remove("General Dentistry")
catagory_50.remove("Pet Services")
catagory_50.remove("Home Decor")
catagory_50.remove("Furniture Stores")
catagory_50.remove("Flowers & Gifts")
catagory_50.remove("Tires")
catagory_50.remove("Oil Change Stations")


# COMMAND ----------

catagory_50

# COMMAND ----------

#FILTER THE DATA BASED ON LIST

business = business.filter(business.cat1.isin(catagory_50)|business.cat2.isin(catagory_50)|business.cat3.isin(catagory_50)|business.cat5.isin(catagory_50)|business.cat5.isin(catagory_50))
business.display()

# COMMAND ----------

Restaurant_df=business
#####################
display(Restaurant_df)

# COMMAND ----------

# Check null values
Restaurant_df.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in Restaurant_df.columns]).display()

# COMMAND ----------

# Calculate percent of missing data per column
restaurant_missing_percent = Restaurant_df.agg(*[(1 - (count(c)/count("*"))).alias(c) for c in Restaurant_df.columns])
restaurant_missing_percent.toPandas()

# COMMAND ----------

cols = restaurant_missing_percent.columns

restaurant_missing = restaurant_missing_percent.withColumn("a",explode(arrays_zip(array(*map(lambda x: lit(x), cols)), array(*cols), )))\
    .select("a.*")\
    .toDF(*['Variable','Value'])
    
restaurant_missing.display()

# COMMAND ----------

# Check number of unique business
Restaurant_df.agg(countDistinct(col("business_id")).alias("count_businessid")).show()

# COMMAND ----------

# https://stackoverflow.com/questions/52633916/outlier-detection-in-pyspark
# Check outliers
features = ['review_count', 'stars']
bounds = {c: dict(
        zip(["q1", "q3"], Restaurant_df.approxQuantile(c, [0.25, 0.75], 0))) for c in features}

for c in bounds:
    iqr = bounds[c]['q3'] - bounds[c]['q1']
    bounds[c]['lower'] = bounds[c]['q1'] - (iqr * 1.5)
    bounds[c]['upper'] = bounds[c]['q3'] + (iqr * 1.5)
print(bounds)

# COMMAND ----------

Restaurant_df.describe(features).show()

# COMMAND ----------

# Visulaize the distribution of review_count and stars
Restaurant_df.groupby("review_count").count().display()

# COMMAND ----------

Restaurant_df.groupby("stars").count().display()

# COMMAND ----------

#Extract the opening hour


Restaurant_df = Restaurant_df.withColumn('Opens_morning5', split(Restaurant_df['hours_Friday'], '-').getItem(0))
Restaurant_df = Restaurant_df.withColumn('Opens_morning5', split(Restaurant_df['Opens_morning5'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_morning5',Restaurant_df.Opens_morning5.cast("short"))

###

Restaurant_df = Restaurant_df.withColumn('Opens_morning1', split(Restaurant_df['hours_Monday'], '-').getItem(0))
Restaurant_df = Restaurant_df.withColumn('Opens_morning1', split(Restaurant_df['Opens_morning1'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_morning1',Restaurant_df.Opens_morning1.cast("short"))

####


Restaurant_df = Restaurant_df.withColumn('Opens_morning6', split(Restaurant_df['hours_Saturday'], '-').getItem(0))
Restaurant_df = Restaurant_df.withColumn('Opens_morning6', split(Restaurant_df['Opens_morning6'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_morning6',Restaurant_df.Opens_morning6.cast("short"))

###

Restaurant_df = Restaurant_df.withColumn('Opens_morning7', split(Restaurant_df['hours_Sunday'], '-').getItem(0))
Restaurant_df = Restaurant_df.withColumn('Opens_morning7', split(Restaurant_df['Opens_morning7'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_morning7',Restaurant_df.Opens_morning7.cast("short"))

###


Restaurant_df = Restaurant_df.withColumn('Opens_morning4', split(Restaurant_df['hours_Thursday'], '-').getItem(0))
Restaurant_df = Restaurant_df.withColumn('Opens_morning4', split(Restaurant_df['Opens_morning4'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_morning4',Restaurant_df.Opens_morning4.cast("short"))

####


Restaurant_df = Restaurant_df.withColumn('Opens_morning2', split(Restaurant_df['hours_Tuesday'], '-').getItem(0))
Restaurant_df = Restaurant_df.withColumn('Opens_morning2', split(Restaurant_df['Opens_morning2'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_morning2',Restaurant_df.Opens_morning2.cast("short"))

###


Restaurant_df = Restaurant_df.withColumn('Opens_morning3', split(Restaurant_df['hours_Wednesday'], '-').getItem(0))
Restaurant_df = Restaurant_df.withColumn('Opens_morning3', split(Restaurant_df['Opens_morning3'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_morning3',Restaurant_df.Opens_morning3.cast("short"))

###

# COMMAND ----------

#Create open on weekdays and opens weekend
from pyspark.sql.functions import when
Restaurant_df=Restaurant_df.withColumn("Opens_All_Weekdays",when((Restaurant_df.Opens_morning1.isNull().cast('int')+Restaurant_df.Opens_morning2.isNull().cast('int')+Restaurant_df.Opens_morning3.isNull().cast('int')+Restaurant_df.Opens_morning4.isNull().cast('int')+Restaurant_df.Opens_morning5.isNull().cast('int')>0),lit(0)).otherwise(lit(1)))


Restaurant_df=Restaurant_df.withColumn("Opens_Weekends_sunday",when((Restaurant_df.Opens_morning6.isNull().cast('int')+Restaurant_df.Opens_morning7.isNull().cast('int')>0),lit(0)).otherwise(lit(1)))


Restaurant_df=Restaurant_df.withColumn("Days_open",7-(Restaurant_df.Opens_morning1.isNull().cast('int')+Restaurant_df.Opens_morning2.isNull().cast('int')+Restaurant_df.Opens_morning3.isNull().cast('int')+Restaurant_df.Opens_morning4.isNull().cast('int')+Restaurant_df.Opens_morning5.isNull().cast('int')+Restaurant_df.Opens_morning6.isNull().cast('int')+Restaurant_df.Opens_morning7.isNull().cast('int')))


Restaurant_df=Restaurant_df.withColumn("no_info_open",when((Restaurant_df.Days_open ==0),lit(1)).otherwise(lit(0)))

Restaurant_df=Restaurant_df.withColumn("Opens_morning1",when((Restaurant_df.Opens_morning1 <11),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_morning2",when((Restaurant_df.Opens_morning2 <11),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_morning3",when((Restaurant_df.Opens_morning3 <11),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_morning4",when((Restaurant_df.Opens_morning4 <11),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_morning5",when((Restaurant_df.Opens_morning5 <11),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_morning6",when((Restaurant_df.Opens_morning6 <11),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_morning7",when((Restaurant_df.Opens_morning7 <11),lit(1)).otherwise(0))


# 

Restaurant_df=Restaurant_df.withColumn("Opened_morning",(Restaurant_df.Opens_morning1 + Restaurant_df.Opens_morning2 + Restaurant_df.Opens_morning3 + Restaurant_df.Opens_morning4 + Restaurant_df.Opens_morning5 + Restaurant_df.Opens_morning6 + Restaurant_df.Opens_morning7)/Restaurant_df.Days_open)

#fill nulls on opens morning replace with 0


# COMMAND ----------


#extract the colsing hours per day
Restaurant_df = Restaurant_df.withColumn('Opens_night1', split(Restaurant_df['hours_Monday'], '-').getItem(1))
Restaurant_df = Restaurant_df.withColumn('Opens_night1', split(Restaurant_df['Opens_night1'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_night1',Restaurant_df.Opens_night1.cast("short"))

Restaurant_df = Restaurant_df.withColumn('Opens_night2', split(Restaurant_df['hours_Tuesday'], '-').getItem(1))
Restaurant_df = Restaurant_df.withColumn('Opens_night2', split(Restaurant_df['Opens_night2'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_night2',Restaurant_df.Opens_night2.cast("short"))

Restaurant_df = Restaurant_df.withColumn('Opens_night3', split(Restaurant_df['hours_Wednesday'], '-').getItem(1))
Restaurant_df = Restaurant_df.withColumn('Opens_night3', split(Restaurant_df['Opens_night3'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_night3',Restaurant_df.Opens_night3.cast("short"))

Restaurant_df = Restaurant_df.withColumn('Opens_night4', split(Restaurant_df['hours_Thursday'], '-').getItem(1))
Restaurant_df = Restaurant_df.withColumn('Opens_night4', split(Restaurant_df['Opens_night4'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_night4',Restaurant_df.Opens_night4.cast("short"))

Restaurant_df = Restaurant_df.withColumn('Opens_night5', split(Restaurant_df['hours_Friday'], '-').getItem(1))
Restaurant_df = Restaurant_df.withColumn('Opens_night5', split(Restaurant_df['Opens_night5'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_night5',Restaurant_df.Opens_night5.cast("short"))

Restaurant_df = Restaurant_df.withColumn('Opens_night6', split(Restaurant_df['hours_Saturday'], '-').getItem(1))
Restaurant_df = Restaurant_df.withColumn('Opens_night6', split(Restaurant_df['Opens_night6'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_night6',Restaurant_df.Opens_night6.cast("short"))

Restaurant_df = Restaurant_df.withColumn('Opens_night7', split(Restaurant_df['hours_Sunday'], '-').getItem(1))
Restaurant_df = Restaurant_df.withColumn('Opens_night7', split(Restaurant_df['Opens_night7'], ':').getItem(0))
Restaurant_df= Restaurant_df.withColumn('Opens_night7',Restaurant_df.Opens_night7.cast("short"))

Restaurant_df=Restaurant_df.withColumn("Opens_night1",when(((Restaurant_df.Opens_night1 <4) | (Restaurant_df.Opens_night1 >18)),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_night2",when(((Restaurant_df.Opens_night2 <4) | (Restaurant_df.Opens_night2 >18)),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_night3",when(((Restaurant_df.Opens_night3 <4) | (Restaurant_df.Opens_night3 >18)),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_night4",when(((Restaurant_df.Opens_night4 <4) | (Restaurant_df.Opens_night4 >18)),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_night5",when(((Restaurant_df.Opens_night5 <4) | (Restaurant_df.Opens_night5 >18)),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_night6",when(((Restaurant_df.Opens_night6 <4) | (Restaurant_df.Opens_night6 >18)),lit(1)).otherwise(0))
Restaurant_df=Restaurant_df.withColumn("Opens_night7",when(((Restaurant_df.Opens_night7 <4) | (Restaurant_df.Opens_night7 >18)),lit(1)).otherwise(0))

Restaurant_df=Restaurant_df.withColumn("Opened_night",(Restaurant_df.Opens_night1 + Restaurant_df.Opens_night2 + Restaurant_df.Opens_night3 + Restaurant_df.Opens_night4 + Restaurant_df.Opens_night5 + Restaurant_df.Opens_night6 + Restaurant_df.Opens_night7)/Restaurant_df.Days_open)

# COMMAND ----------

## drop multiple columns starts with a string 
list=Restaurant_df.columns
columns_to_drop = [i for i in list if i.startswith('Opens_morning')]
Restaurant_df = Restaurant_df.drop(*columns_to_drop)

list=Restaurant_df.columns
columns_to_drop = [i for i in list if i.startswith('Opens_night')]
Restaurant_df = Restaurant_df.drop(*columns_to_drop)

list=Restaurant_df.columns
columns_to_drop = [i for i in list if i.startswith('Opens_night')]
Restaurant_df = Restaurant_df.drop(*columns_to_drop)

# COMMAND ----------

Restaurant_df= Restaurant_df.na.fill(value=0,subset=["Opened_night"])
Restaurant_df= Restaurant_df.na.fill(value=0,subset=["Opened_morning"])

# COMMAND ----------


display(Restaurant_df)

# COMMAND ----------

# DBTITLE 1,Table: review
#Read data 
review = spark.read.format("json")\
    .option("header","true")\
    .option("inferSchema","true")\
    .load(review_path)
review.show(5)
review.printSchema()

# COMMAND ----------

#Inspect the table
review.describe().show()

# COMMAND ----------

# Missing values
review.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in review.columns]).show()

# COMMAND ----------

# Check duplicates
review.agg(count("business_id").alias("CountID"),
         countDistinct("business_id").alias("CountDistinctID")).show()

# COMMAND ----------

# Count distinct user_id
review.agg(countDistinct(col("user_id")).alias("count_userid")).show()

# Calculate number of stars
review.groupby(review['stars']).count().sort(desc("count")).show()

# COMMAND ----------

# Extracting year from the date
review_temp = review.withColumn('year',year('date'))

# Extracting month from the date
review_temp = review_temp.withColumn('month',month('date'))

# Extracting day from the date
review_temp = review_temp.withColumn('day',dayofmonth('date'))

# COMMAND ----------

# Calculating the distinct business_id by the year
review_temp.groupBy("year").agg(countDistinct("business_id")).sort("count(business_id)").show()

# COMMAND ----------

# Calculate total number of votes received per business_id
review = review.withColumn('total_votes', col('cool') + col('funny') + col('useful'))
review.show(3)

# COMMAND ----------

# Calculate recency
review = review.withColumn('recency', datediff(lit('2020-03-31'), col('date')))
review.show(5)  

# COMMAND ----------

# Create new variables per business_id
review_df = review.groupby('business_id').agg(round(mean('stars'),2).alias("avg_stars"), countDistinct('review_id').alias('nbr_reviews'), sum('cool').alias('total_cool'),\
                                             sum('funny').alias('total_funny'), sum('useful').alias('total_useful'),\
                                             sum('total_votes').alias('total_votes'), round(mean('total_votes'),2).alias('avg_votes'),\
                                             min('recency').alias('recent_review'))

review_df = review_df.withColumn('ratio_cool', round(col('total_cool')/col('total_votes'), 2))\
    .withColumn('ratio_funny', round(col('total_funny')/col('total_votes'), 2))\
    .withColumn('ratio_useful', round(col('total_useful')/col('total_votes'), 2))
review_df.show(5)

# COMMAND ----------

# DBTITLE 1,Table: covid
# Read data
covid = spark.read.format("json")\
    .option("header","true")\
    .option("inferSchema","true")\
    .load(covid_path)
covid.show(5)
covid.printSchema()

# COMMAND ----------

# Missing values
covid.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in covid.columns]).show()

# COMMAND ----------

# Check duplicates
covid.agg(count("business_id").alias("CountID"),
         countDistinct("business_id").alias("CountDistinctID")).show()

# COMMAND ----------

# Drop duplicates
covid_filter = covid.dropDuplicates(["business_id", "delivery or takeout"])

# COMMAND ----------

# Select only business_id and delivery or takeout
covid_df = covid_filter.select("business_id", "delivery or takeout")
covid_df.count()

# COMMAND ----------

from pyspark.sql.types import *
covid_df =covid_df.withColumn("delivery or takeout",col("delivery or takeout").cast(BooleanType()).cast("int"))
covid_df.describe()

# COMMAND ----------

# Create final table
data = Restaurant_df.join(review_df, "business_id", "left")\
    .join(checkin_df, "business_id", "left")\
    .join(tip_df,"business_id", "left")\
    .join(covid_df, "business_id", "left")
data.count()
data.display()

# COMMAND ----------

# Drop columns start with hours
list = data.columns
columns_to_drop = [i for i in list if i.startswith('hours')]
data = data.drop(*columns_to_drop)

# COMMAND ----------

# Check null values
data.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in data.columns]).display()

# COMMAND ----------

# Calculate percent of missing data per column
data_missing_percent = data.agg(*[(1 - (count(c)/count("*"))).alias(c) for c in data.columns])
data_missing_percent.toPandas()

# COMMAND ----------

cols = data_missing_percent.columns

data_missing = data_missing_percent.withColumn("a",explode(arrays_zip(array(*map(lambda x: lit(x), cols)), array(*cols), )))\
     .select("a.*")\
     .toDF(*['Variable','Value'])
    
data_missing.display()

# COMMAND ----------

 # Drop columns with missing percent more than 40% except 'cat5'
drop_columns = data_missing.filter(col("Value") > 0.40).select(col("Variable"))

# Convert pyspark dataframe column to list 
drop_list = drop_columns.select("Variable").rdd.map(lambda x : x[0]).collect()
#del drop_list("cat4", "cat5")
#drop_list.show()

# COMMAND ----------

data = data.drop(*drop_list)
data.display()

# COMMAND ----------

# ## gather the distinct values of 'state'
distinct_values = data.select("state")\
    .distinct()\
    .rdd\
    .flatMap(lambda x: x).collect()

# COMMAND ----------

#  for each of the gathered values create a new column
for distinct_value in distinct_values:
    function = udf(lambda item:
                   1 if item == distinct_value else 0, IntegerType())
    new_column_name = "state"+'_'+distinct_value
    data = data.withColumn(new_column_name, function(col("state")))

# COMMAND ----------

#rename the attributes_BusinessAcceptsCreditCards column values
data = data.withColumn("attributes_BusinessAcceptsCreditCards", when(data.attributes_BusinessAcceptsCreditCards == "True","Yes") \
      .when(data.attributes_BusinessAcceptsCreditCards == "False","No") \
      #.when(data.attributes_BusinessAcceptsCreditCards == "","Missing") \ 
      .otherwise('Missing'))

# COMMAND ----------

# ## gather the distinct values of 'attributes_BusinessAcceptsCreditCards'
distinct_values = data.select("attributes_BusinessAcceptsCreditCards")\
    .distinct()\
    .rdd\
    .flatMap(lambda x: x).collect()

# COMMAND ----------

# ## for each of the gathered values create a new column
for distinct_value in distinct_values:
    function = udf(lambda item:
    1 if item == distinct_value else 0,
    IntegerType())
    new_column_name = "BusinessAcceptsCreditCards"+'_'+distinct_value
    data = data.withColumn(new_column_name, function(col("attributes_BusinessAcceptsCreditCards")))

# COMMAND ----------

data= data.na.fill(value='0',subset=["attributes_RestaurantsPriceRange2"])

# COMMAND ----------

data.display()

# COMMAND ----------

# ## gather the distinct values of 'attributes_RestaurantsPriceRange2'
distinct_values = data.select("attributes_RestaurantsPriceRange2")\
    .distinct()\
    .rdd\
    .flatMap(lambda x: x).collect()

# COMMAND ----------

# ## for each of the gathered values create a new column
for distinct_value in distinct_values:
    function = udf(lambda item:
    1 if item == distinct_value else 0,
    IntegerType())
    new_column_name = "RestaurantsPriceRange"+'_'+distinct_value
    data = data.withColumn(new_column_name, function(col("attributes_RestaurantsPriceRange2")))

# COMMAND ----------

data.display()

# COMMAND ----------

data.printSchema()

# COMMAND ----------

#get the most used categories based on those that started doing deliveries
categories_target = data.filter(col("delivery or takeout") == 1 )
categories_target = categories_target.select(explode(split('categories', ',')).alias('category'))
categories_target = categories_target.withColumn('category', trim(categories_target.category))
categories_target = categories_target.groupby("category").agg(count("category").alias("frecuency")).sort(col("frecuency").desc()).limit(10)
categories_target.display()

# COMMAND ----------

categories_target = categories_target.toPandas()
categories_target.plot.bar(x="category", y= "frecuency")

# COMMAND ----------

#encode categories using the WOE
import category_encoders as ce
datapy = data.toPandas()
X = datapy.loc[:, datapy.columns!='delivery or takeout']
y = datapy["delivery or takeout"]
encoder = ce.WOEEncoder(cols=["cat1","cat2","cat3"])
encoder.fit(X,y)
datapy = encoder.transform(X)

# COMMAND ----------

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
data = spark.createDataFrame(datapy)
data = data.join(covid_df, "business_id", "left")

# COMMAND ----------

# Check null values
data.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in data.columns]).display()

# COMMAND ----------

data.columns

# COMMAND ----------

data = data.drop("address","state","attributes_BusinessParking", "attributes_RestaurantsPriceRange2","attributes_RestaurantsDelivery","attributes_RestaurantsTakeOut","attributes_BusinessAcceptsCreditCards","Categories","City","latitude","longitude","name","postal_code","BusinessAcceptsCreditCards_Missing", "RestaurantsPriceRange_None", "RestaurantsPriceRange_0","pseudo_target")

# COMMAND ----------

data = data.withColumnRenamed("delivery or takeout","delivery_or_takeout")

# COMMAND ----------

#final base table
data.display()

# COMMAND ----------

#check the columns of the final table
data.columns

# COMMAND ----------

# DBTITLE 1,Create train and test set
#Create a train and test set with a 70% train, 30% test split
train, test = data.randomSplit([0.7, 0.3],seed=42)

print(train.count())
print(test.count())

# COMMAND ----------

train.display()

# COMMAND ----------

train.printSchema()

# COMMAND ----------

#Check missing values
train.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in train.columns]).display()

# COMMAND ----------

test.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in test.columns]).display()

# COMMAND ----------

#change null values to 0 to find the average per column in train set
train_df = train.fillna(value=0, subset=['ratio_cool','ratio_funny','ratio_useful','days_since last_check','nbr_checkin','compliment_count','days_since last_tip'])


# COMMAND ----------

#fill the na values with mean for the columns in the train set
df_mean1 = train_df.select(round(mean(col('ratio_cool')),2).alias('avg1')).collect()
avg1 = df_mean1[0]['avg1']

df_mean2 = train_df.select(round(mean(col('ratio_funny')),2).alias('avg2')).collect()
avg2 = df_mean2[0]['avg2']

df_mean3 = train_df.select(round(mean(col('ratio_useful')),2).alias('avg3')).collect()
avg3 = df_mean3[0]['avg3']

df_mean4 = train_df.select(round(mean(col('nbr_checkin')),2).alias('avg4')).collect()
avg4 = df_mean4[0]['avg4']

df_mean5 = train_df.select(round(mean(col('days_since last_check')),2).alias('avg5')).collect()
avg5 = df_mean5[0]['avg5']

df_mean6 = train_df.select(round(mean(col('compliment_count')),2).alias('avg6')).collect()
avg6 = df_mean6[0]['avg6']

df_mean7 = train_df.select(round(mean(col('days_since last_tip')),2).alias('avg7')).collect()
avg7 = df_mean7[0]['avg7']


train = train.fillna(value=avg2, subset=['ratio_cool'])
train = train.fillna(value=avg2, subset=['ratio_funny'])
train = train.fillna(value=avg3, subset=['ratio_useful'])
train = train.fillna(value=avg4, subset=['nbr_checkin'])
train = train.fillna(value=avg5, subset=['days_since last_check'])
train = train.fillna(value=avg6, subset=['compliment_count'])
train = train.fillna(value=avg7, subset=['days_since last_tip'])

# COMMAND ----------

train.display()

# COMMAND ----------

#change null values to 0 to find the average per column in test set
test_df = test.fillna(value=0, subset=['ratio_cool','ratio_funny','ratio_useful','days_since last_check', 'nbr_checkin','compliment_count','days_since last_tip'])

# COMMAND ----------

#fill the na values with mean for the columns in the test set
df_mean1 = test_df.select(round(mean(col('ratio_cool')),2).alias('avg1')).collect()
avg1 = df_mean1[0]['avg1']

df_mean2 = test_df.select(round(mean(col('ratio_funny')),2).alias('avg2')).collect()
avg2 = df_mean2[0]['avg2']

df_mean3 = test_df.select(round(mean(col('ratio_useful')),2).alias('avg3')).collect()
avg3 = df_mean3[0]['avg3']

df_mean4 = test_df.select(round(mean(col('nbr_checkin')),2).alias('avg4')).collect()
avg4 = df_mean4[0]['avg4']

df_mean5 = test_df.select(round(mean(col('days_since last_check')),2).alias('avg5')).collect()
avg5 = df_mean5[0]['avg5']

df_mean5 = test_df.select(round(mean(col('compliment_count')),2).alias('avg6')).collect()
avg6 = df_mean5[0]['avg6']

df_mean6 = test_df.select(round(mean(col('days_since last_tip')),2).alias('avg7')).collect()
avg7 = df_mean6[0]['avg7']


test = test.fillna(value=avg2, subset=['ratio_cool'])
test = test.fillna(value=avg2, subset=['ratio_funny'])
test = test.fillna(value=avg3, subset=['ratio_useful'])
test = test.fillna(value=avg4, subset=['nbr_checkin'])
test = test.fillna(value=avg5, subset=['days_since last_check'])
test = test.fillna(value=avg6, subset=['compliment_count'])
test = test.fillna(value=avg7, subset=['days_since last_tip'])

# COMMAND ----------

test.display()

# COMMAND ----------

# Check missing values for train set
train.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in train.columns]).display()
train.printSchema()

# COMMAND ----------

# Check missing values for test set
test.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in test.columns]).display()
test.printSchema()

# COMMAND ----------

# modelling
#Transform the tables in a table of label, features format
from pyspark.ml.feature import RFormula


train = RFormula(formula="delivery_or_takeout ~ . - business_id").fit(train).transform(train)
test = RFormula(formula="delivery_or_takeout ~ . - business_id").fit(test).transform(test)

print("train nobs: " + str(train.count()))
print("test nobs: " + str(test.count()))

# COMMAND ----------

#Train a Logistic Regression model
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

#parameter-tuning
#Spark offers two options for performing hyperparameter tuning automatically:

#1. TrainValidationSplit: randomly split data in 2 groups
from pyspark.ml.tuning import TrainValidationSplit

#2. CrossValidator: k-fold cross-validation by splitting the data into k non-overlapping, randomly partitioned folds
from pyspark.ml.tuning import CrossValidator

#First method is good for quick model evaluations, 2nd method is recommended for a more rigorous model evaluation.#Hyperparameter tuning for different hyperparameter values of LR (aka model selection)
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Define pipeline
lr = LogisticRegression()
pipeline = Pipeline().setStages([lr])

#Set param grid
params = ParamGridBuilder()\
  .addGrid(lr.regParam, [0.1, 0.01])\
  .addGrid(lr.maxIter, [50, 100,150])\
  .build()

#Evaluator: uses max(AUC) by default to get the final model
evaluator = BinaryClassificationEvaluator()
#Check params through: evaluator.explainParams()

#Cross-validation of entire pipeline
cv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEstimatorParamMaps(params)\
  .setEvaluator(evaluator)\
  .setNumFolds(10) # Here: 10-fold cross validation

#Run cross-validation, and choose the best set of parameters.
#Spark automatically saves the best model in cvModel.
cvModel = cv.fit(train)

#FYI: ignore warning of databricks:
"""
/databricks/spark/python/pyspark/ml/util.py:791: UserWarning: Can not find mlflow. To enable mlflow logging, install MLflow library from PyPi.
  warnings.warn(_MLflowInstrumentation._NO_MLFLOW_WARNING)
"""

# COMMAND ----------

#Get best tuned parameters of pipeline
cvBestPipeline = cvModel.bestModel
cvBestLRModel = cvBestPipeline.stages[-1]._java_obj.parent() #the stages function refers to the stage in the pipelinemodel

print("Best LR model:")
print("** regParam: " + str(cvBestLRModel.getRegParam()))
print("** maxIter: " + str(cvBestLRModel.getMaxIter()))

# COMMAND ----------

preds = cvModel.transform(test)\
  .select("prediction", "delivery_or_takeout")
preds.show(10)

# COMMAND ----------

#Get model performance on test set
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = preds.rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR) #area under precision/recall curve
print(metrics.areaUnderROC)#area under Receiver Operating Characteristic curve

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

#Cast a DF of predictions to an RDD to access RDD methods of MulticlassMetrics
preds_labels = cvModel.transform(test)\
  .select("prediction", "delivery_or_takeout")\
  .rdd.map(lambda x: (float(x[0]), float(x[1])))

metrics = MulticlassMetrics(preds_labels)

print("accuracy = %s" % metrics.accuracy)

# COMMAND ----------

#Random forests
#Exercise: Train a RandomForest model and tune the number of trees for values [150, 300, 500]
#Hint: analogous to buidling a LR model (see above)
from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

#Exercise: Train a RandomForest model and tune the number of trees for values [150, 300, 500]
from pyspark.ml.classification import RandomForestClassifier

#Define pipeline
rfc = RandomForestClassifier()
rfPipe = Pipeline().setStages([rfc])

#Set param grid
rfParams = ParamGridBuilder()\
  .addGrid(rfc.numTrees, [150, 200,300 ])\
  .build()

rfCv = CrossValidator()\
  .setEstimator(rfPipe)\
  .setEstimatorParamMaps(rfParams)\
  .setEvaluator(BinaryClassificationEvaluator())\
  .setNumFolds(10) # Here: 5-fold cross validation

#Run cross-validation, and choose the best set of parameters.
rfcModel = rfCv.fit(train)

# COMMAND ----------

#Get predictions on the test set
preds = rfcModel.transform(test)
preds.show(5)

# COMMAND ----------

#Get model accuracy
print("accuracy: " + str(evaluator.evaluate(preds)))

#Get AUC
metrics = BinaryClassificationMetrics(preds.select('prediction','delivery_or_takeout').rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("AUC: " + str(metrics.areaUnderROC))

# COMMAND ----------



# COMMAND ----------

#Select the best RF model
rfcBestModel = rfcModel.bestModel.stages[-1] #-1 means "get last stage in the pipeline"

# COMMAND ----------

#Get tuned number of trees
rfcBestModel.getNumTrees

# COMMAND ----------

#Get feature importances
rfcBestModel.featureImportances

# COMMAND ----------

#Prettify feature importances
import pandas as pd
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
ExtractFeatureImp(rfcBestModel.featureImportances, train, "features").head(10)

# COMMAND ----------

                                                                                                                                                                                                                                                                                                                            
