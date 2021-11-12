import os

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def show_image_as_array(image_as_array):
  x = np.reshape(image_as_array,(224,224,3))
  img = keras.preprocessing.image.array_to_img(x)
  imgplot = plt.imshow(img)
  return x.shape


def get_files_list(path):
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.jpg']
    #files = files[:2048]
    return files



def load_images(path,spark):
    """
    loads all images from path to Spark Dataframe
    images are in jpg format
    """
    # to do: create Dataframe
    raise NotImplementedError

    files = get_files_list(path)
    for file in files:
      img = Image.open(file)
      img = img.resize([224, 224])
      image_array = np.asarray( img, dtype="float32" ).reshape([224*224*3])
      #add image to Dataframe
    
    pass