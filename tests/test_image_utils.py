import pytest

import numpy as np
from PIL import Image

from pyspark.ml.linalg import Vectors

from distrib_resnet.image_utils import load_images_to_list, show_image_as_array
from distrib_resnet.image_utils import images_to_df
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


def test_load_images_to_list():
    path = "./images/"
    l = load_images_to_list(path)
    assert len(l) == 6 and l[4][1] == 153.


def test_load_images_to_dataframe(spark):
    path = "./images/"
    l = load_images_to_list(path)
    df = images_to_df(l,spark)
    assert df.count() == 6