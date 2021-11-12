# """
# Definitions for all the spark tests
# """
import pytest
import numpy as np

import findspark
findspark.init()

from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.sql.types import StructField, ArrayType, StructType, DoubleType, StringType



@pytest.fixture(scope="session")
def spark():
    """
    Create spark session on test session level and  access spark api from any test int he folder
    """
    spark_conf = SparkConf(loadDefaults=True) \
        .set("spark.sql.session.timeZone", "UTC")
    return SparkSession \
        .builder \
        .config(conf=spark_conf) \
        .getOrCreate()

@pytest.fixture(scope="session")
def image_as_numpy_dataframe_schema():
    """
    Returns schema for data frame with images
    """
    schema = StructType().add("image_as_numpy", ArrayType(DoubleType(), False), False)
    return schema
