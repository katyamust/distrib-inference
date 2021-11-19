import pytest

import numpy as np
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from distrib_resnet import predict_batch


@pytest.fixture()
def spark_df(spark):
    """
    Generate small sample dataframe for test
    """
    df = np.concatenate([np.arange(5),
                     2*np.arange(5),
                     3*np.arange(5),
                     4*np.arange(5),
                     5*np.arange(5)]).reshape(5,-1)
    dff = map(lambda x: (int(x[1]), Vectors.dense(x[1:])), df)

    mydf = spark.createDataFrame(dff,schema=["label", "features"])
    return mydf

def test_predict_batch(spark_df):

    predictions_df = spark_df.select(col('features'), predict_batch(col('features')).alias("prediction"))
    predictions_df.write.mode("noop").parquet("")

    assert True