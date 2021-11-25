import pytest

import numpy as np
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from distrib_resnet.predict_batch import predict_batch


@pytest.fixture()
def spark_df(spark):
    """
    Generate small sample dataframe for test
    """
    lengh = 10
    nrows = 5
    df = np.concatenate([np.arange(lengh),
                     2*np.arange(lengh),
                     3*np.arange(lengh),
                     4*np.arange(lengh),
                     5*np.arange(lengh)]).reshape(nrows,-1)
    dff = map(lambda x: (1,Vectors.dense(x)), df)
    #dff = map(lambda x: (int(x[1]), Vectors.dense(x[1:])), df)

    mydf = spark.createDataFrame(dff,schema=["_", "features"])
    return mydf

def test_predict_batch(spark_df):

    nrows = spark_df.count()
    predictions_df = spark_df.select(col('features'), predict_batch(col('features')).alias("prediction"))
    nprows = predictions_df.count()

    assert nrows == nprows