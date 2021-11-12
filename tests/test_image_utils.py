import pytest

import numpy as np
import pandas as pd
from PIL import Image

from pyspark.ml.linalg import Vectors

from distrib_resnet.image_utils import load_images, show_image_as_array
from distrib_resnet.image_utils import get_files_list

def test_show_image_as_array():
    image_file  = "./images/nilo.jpg"
    img = Image.open(image_file)
    img = img.resize([224, 224])
    image_array = np.asarray( img, dtype="float32" ) #.reshape([224*224*3])
    image_shape = show_image_as_array(image_array)
    assert image_shape == (224,224,3)

def test_get_files_list():
    path = "./images/"
    files = get_files_list("./images/")
    assert len(files) == 6 and files[4] == "./images/elephant.jpg"


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


def test_load_images_to_dataframe(spark,spark_df):
    path = "./images/"
    df_raw = load_images(path,spark)
    rows = df_raw.count()
    cols = len(df_raw.columns)
    assert (rows, cols) == (spark_df.count(),len(spark_df.columns)) 