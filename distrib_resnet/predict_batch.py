import torch

from pyspark.sql.types import ArrayType, FloatType
from distrib_resnet.image_dataset import ImageDataset
from distrib_resnet.load_model import get_broadcasted_model_for_eval
import pandas as pd

@pandas_udf(ArrayType(FloatType()))
def predict_batch(input_data: pd.Series, device) -> pd.Series:
  mydata  = ImageDataset(input_data)
  loader = torch.utils.data.DataLoader(mydata, batch_size=ImageDataset.get_batch_size(), num_workers=1)
  model = get_broadcasted_model_for_eval()
  model.to(device)
  all_predictions = []
  with torch.no_grad():
    for batch in loader:
      predictions = list(model(batch.to(device)).cpu().numpy())
      for prediction in predictions:
        all_predictions.append(prediction)
  return pd.Series(all_predictions)
