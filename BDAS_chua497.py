
# coding: utf-8

# # basic

# In[1]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('BDAS_chua497').getOrCreate()


# In[2]:


data = spark.read.csv("who-revised.csv",inferSchema=True,header=True)


# # 2.3

# In[3]:


data.printSchema()


# In[4]:


data.head()


# In[5]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from pyspark.sql.types import DoubleType
#data.columns[3]
#data.select(data.columns[3]).show()
for i in range(2, 20):
    data = data.withColumn(data.columns[i], data[data.columns[i]].cast(DoubleType()))

data.printSchema()

y = data.select(data.columns[3]).collect()
for i in range(2, 20):
    plt.subplot(6,  3,  i - 1)  
    x = data.select(data.columns[i]).collect()
    plt.scatter(x, y)
    plt.title(data.columns[i])
plt.show()


# # 2.4

# In[6]:


data.count()
print("Missing value:")
for i in range(1, 20):
    row = "%s is null" %(data.columns[i])
    print("%s : %d/%d" %(data.columns[i], data.where(row).count(), data.count()))


# In[7]:


x = data.select(data.columns[0]).collect()
get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(3, 20):
    plt.subplot(6,  3,  i - 2)  
    y = data.select(data.columns[i]).collect()
    plt.scatter(x, y, color='green', marker='o', edgecolor='black', alpha=0.5)
    plt.title(data.columns[i])
plt.show()


# # 3.3

# In[8]:


data_claen = data.where("Hepatitis_B > 10 and Polio > 10 and Diphtheria > 10 and Income_composition_of_resources > 0.1")
x = data_claen.select(data_claen.columns[0]).collect()
get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(3, 20):
    plt.subplot(6,  3,  i - 2)  
    y = data_claen.select(data_claen.columns[i]).collect()
    plt.scatter(x, y, color='green', marker='o', edgecolor='black', alpha=0.5)
    plt.title(data_claen.columns[i])
plt.show()


# In[9]:


print(data_claen.count())
data_claen2 = data_claen.na.drop()
print(data_claen2.count())


# # 3.4

# In[10]:


construct_text = data.withColumn("GDP_Population", data["GDP"] + data["Population"])
construct_text.printSchema()
construct_text.select("GDP", "Population", "GDP_Population").show()


# # 3.5

# In[11]:


Integrate_text = data.unionAll(data)
Integrate_text.printSchema()
print(data.count())
print(Integrate_text.count())


# # 4.1

# In[12]:


data_reduce = data_claen2.drop("_c0", "Country", "Population")
data_reduce.printSchema()


# # 4.2

# In[13]:


print(data_reduce.select("Schooling").rdd.max()[0])
print(data_reduce.select("Schooling").rdd.min()[0])
print(data_reduce.select("GDP").rdd.max()[0])
print(data_reduce.select("GDP").rdd.min()[0])


# In[14]:


import math
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.sql import functions as F

print(data_reduce.head())
for i in range(0, 17):
    if(i == 1):
        continue
    data_reduce = data_reduce.withColumn(data_reduce.columns[i], F.log10(col(data_reduce.columns[i]) + 1))

print(data_reduce.head())


# In[15]:


x = [i for i in range(0, data_reduce.count())]
get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(0, 17):
    plt.subplot(6,  3,  i + 1)  
    y = data_reduce.select(data_reduce.columns[i]).collect()
    plt.scatter(x, y, color='green', marker='o', edgecolor='black', alpha=0.5)
    plt.title(data_reduce.columns[i])
plt.show()


# # 6.3

# In[16]:


data = data_reduce
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols=["Year", "Adult_Mortality", "infant_deaths"
                                          , "Alcohol", "Hepatitis_B", "Measles", "under_five_deaths"
                                          , "Polio", "Total_expenditure", "Diphtheria", "HIV_AIDS"
                                          , "GDP", "thinness__1_19_years", "thinness_5_9_years"
                                          , "Income_composition_of_resources", "Schooling"], outputCol="features")

final_data = vecAssembler.transform(data_reduce).select("features", "Life_expectancy")

final_data.show()


# In[17]:


train_data,test_data = final_data.randomSplit([0.7,0.3])
train_data.describe().show()
test_data.describe().show()


# In[18]:


from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol="Life_expectancy")
lrModel = lr.fit(train_data)
data.printSchema()
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# In[19]:


print( "Intercept: %f" %(lrModel.intercept))
print("Coefficients:")
for i in range(0, 16):
    j = i
    if(i >= 2):
        j = i+1
    print("%s: %f" %(data.columns[j], lrModel.coefficients[i]))


# In[20]:


test_results = lrModel.evaluate(test_data)
test_results.residuals.show(10)
print("RSME: {}".format(test_results.rootMeanSquaredError))


# In[21]:


print("R2: {}".format(test_results.r2))


# In[22]:


final_data.describe().show()


# # 8.2

# In[23]:


x = [i for i in range(0, test_data.count())]
y = test_results.residuals.collect()
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.scatter(x, y, color='green', marker='o', edgecolor='black', alpha=0.5)
plt.show()


# # 8.5

# In[24]:


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
data = final_data
data.show(10)


# In[25]:


(trainingData, testData) = data.randomSplit([0.7, 0.3])
trainingData.describe().show()
testData.describe().show()
rf = RandomForestRegressor(labelCol="Life_expectancy", featuresCol="features")

model_rf = rf.fit(trainingData)
prediction_rf = model_rf.transform(testData)


# In[26]:


prediction_rf.show()


# In[27]:


evaluator = RegressionEvaluator(labelCol="Life_expectancy", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction_rf)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[28]:


print(model_rf)

