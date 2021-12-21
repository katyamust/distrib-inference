# Databricks notebook source
import numpy as np
import torch

from pyspark.ml.linalg import Vectors

from distrib_resnet.image_utils import load_images_to_list, show_image_as_array, get_files_list
from distrib_resnet.image_utils import images_to_df
from distrib_resnet.image_utils import get_files_list

# COMMAND ----------

file_location = "/dbfs/FileStore/tables"
l = get_files_list(file_location)
l = load_images_to_list(file_location)
df = images_to_df(l,spark)
df.show()

# COMMAND ----------

imx = df.collect()[5][1].values
type(imx)
image_shape = show_image_as_array(imx)

# COMMAND ----------

from torchvision import  models

model_state = models.resnet50(pretrained=True).state_dict()
bc_model_state = sc.broadcast(model_state)

# COMMAND ----------

x = imx.reshape(224,224,3)
#x = np.moveaxis(x, -1, 0)
x.shape

# COMMAND ----------

tx = torch.tensor(x)
txx = torch.unsqueeze(tx, 0)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, udf 
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
#from distrib_resnet.image_dataset import ImageDataset
from torchvision import  transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
  def __init__(self, input_arr, transform=None):
    # Make an array of arrays instead of array of lists
    #self.data = np.array(input_arr) # this is array of arrays
    self.data = np.array([np.array(xi) for xi in input_arr])
    self.transform = transform
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index):
    tx = self.data[index]
    x = np.array(tx)
   # x = x.reshape(224,224,3)
    if self.transform is not None:
      x = self.transform(x)
      x = torch.unsqueeze(x, 0)
    return x
  
  @staticmethod
  def get_batch_size():
      return 500

    
@udf(returnType=ArrayType(FloatType()))
def vector_to_array(v):
    # convert column of vectors into column of arrays
    a = v.values.tolist()
    return a

tf_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@pandas_udf(ArrayType(FloatType()))
def predict_batch(input_data: pd.Series) -> pd.Series:
  mydata  = ImageDataset(input_data) #,torch.tensor)
  loader = torch.utils.data.DataLoader(mydata, batch_size=ImageDataset.get_batch_size(), num_workers=1)
  model = models.resnet50(pretrained=True)
  model.load_state_dict(bc_model_state.value)
  model.eval()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  all_predictions = []
  with torch.no_grad():
    for batch in loader:
      #predictions = list(model(batch.to(device)).cpu().numpy())
      predictions = list(batch.to(device).cpu().numpy())
      for prediction in predictions:
        all_predictions.append(prediction)
  return pd.Series(all_predictions)


# COMMAND ----------

from pyspark.sql.types import ArrayType, FloatType

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df_arr_t = df.select(col('features'), vector_to_array(col('features')).alias("array"))

# COMMAND ----------

df_arr_t.show()

# COMMAND ----------

type(df_arr_t.features)

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
#from distrib_resnet.predict_batch import predict_batch
from pyspark.sql.types import ArrayType, FloatType

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df_arr = df.select(col('features'), vector_to_array(col('features')).alias("array"))
predictions_df = df_arr.select(col('array'), predict_batch(col('array')).alias("prediction"))

#predictions_df.display()

# COMMAND ----------

predictions_df.show()

# COMMAND ----------


