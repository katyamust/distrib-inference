import pandas as pd
import numpy as np

import findspark
findspark.init()

from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.linalg import Vectors

spark_conf = SparkConf(loadDefaults=True) \
    .set("spark.sql.session.timeZone", "UTC")
spark = SparkSession \
    .builder \
    .config(conf=spark_conf) \
    .getOrCreate()

#pdf = pd.DataFrame(data=[np.arange(5),np.arange(5)]) #, columns =["image_as_numpy"])
#df_raw = spark.createDataFrame(pdf).show()

#rows = df_raw.count()
#cols = len(df_raw.columns)
#df_raw.show()

df = np.concatenate([np.random.randint(0,2, size=(5)), np.random.randn(5), 3*np.random.randn(5)+2, 6*np.random.randn(5)-2]).reshape(5,-1)

df = np.concatenate([np.arange(5),
                     2*np.arange(5),
                     3*np.arange(5),
                     4*np.arange(5),
                     5*np.arange(5)]).reshape(5,-1)
dff = map(lambda x: (int(x[1]), Vectors.dense(x[1:])), df)

mydf = spark.createDataFrame(dff,schema=["label", "features"])

mydf.show()

print(type(mydf.collect()[0]["features"].values))


print("spark loaded")