import os

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

from pyspark.ml.linalg import Vectors


def show_image_as_array(image_as_array):
  x = np.reshape(image_as_array,(224,224,3))
  img = keras.preprocessing.image.array_to_img(x)
  imgplot = plt.imshow(img)
  return x.shape


def get_files_list(path):
  files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.jpg']
  #files = files[:2048]
  return files


def load_images_to_list(path): 
  """
  reads all images from path, 
  resizes them to 224x224
  convert to numpy arrays,
  returns: list of images reshaped numpy arrays
  """
  files = get_files_list(path)
  image_as_array_list = []
  for file in files:
    img = Image.open(file)
    img = img.resize([224, 224])
    image_as_array = list(np.asarray( img, dtype="float32" ).reshape([224*224*3]))
    image_as_array_list.append(image_as_array)

  return image_as_array_list


def images_to_df(images_list,spark):
  """
  creates Spark Dataframe from 
  list of images as reshaped numpy arrays
  """
  size = len(images_list)
  df = np.concatenate(images_list).reshape(size,-1)
  dff = map(lambda x: (1,Vectors.dense(x)), df)
  mydf = spark.createDataFrame(dff, schema=["_","features"])
  return mydf

