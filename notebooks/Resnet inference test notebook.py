# Databricks notebook source
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import  models, transforms

# COMMAND ----------

model_state = models.resnet50(pretrained=True).state_dict()
bc_model_state = sc.broadcast(model_state)
model = models.resnet50(pretrained=True)
model.load_state_dict(bc_model_state.value)
m = model.eval()

# COMMAND ----------

def show_result(out):
  #
  # Load the file containing the 1,000 labels for the ImageNet dataset classes
  #
  with open('/dbfs/FileStore/tables/imagenet_classes.txt') as f:
      labels = [line.strip() for line in f.readlines()]
  #
  # Find the index (tensor) corresponding to the maximum score in the out tensor.
  # Torch.max function can be used to find the information
  #
  _, index = torch.max(out, 1)
  #
  # Find the score in terms of percentage by using torch.nn.functional.softmax function
  # which normalizes the output to range [0,1] and multiplying by 100
  #
  percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
  #
  # Print the name along with score of the object identified by the model
  #
  print(labels[index[0]], percentage[index[0]].item())
  #
  # Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
  #
  _, indices = torch.sort(out, descending=True)
  print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])

# COMMAND ----------

raw_img = Image.open('/dbfs/FileStore/tables/nilo.jpg').convert('RGB')
img1 = raw_img.resize([256,256]).crop((16,16,256-16,256-16))
img2 = raw_img.resize([224, 224])
#imx = list(np.asarray( img, dtype="float32" ).reshape([224*224*3]))
#img = keras.preprocessing.image.array_to_img(imx)
imgplot = plt.imshow(img1)

# COMMAND ----------

imgplot = plt.imshow(img2)

# COMMAND ----------

# DBTITLE 1,Raw image preprocessing with pytorch
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tgx = preprocess(raw_img)
tgxx = torch.unsqueeze(tgx, 0)
out = model(tgxx)
show_result(out)

# COMMAND ----------

# DBTITLE 1,Manual image preprocessing and convert to tensor with pytorch
imx = np.array(img1,  dtype=np.double )
imx /= 255
print(type(imx))
print(imx.shape)
print(imx)

# COMMAND ----------

# DBTITLE 1,model inference
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tx = preprocess(imx)
txx = torch.unsqueeze(tx.float(), 0)
out = model(txx)
show_result(out)

# COMMAND ----------

print(tgx)

# COMMAND ----------

print(tx)

# COMMAND ----------


