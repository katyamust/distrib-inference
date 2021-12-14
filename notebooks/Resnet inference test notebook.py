# Databricks notebook source
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

# COMMAND ----------

img = Image.open('/dbfs/FileStore/tables/nilo.jpg').convert('RGB')
img = img.resize([224, 224])
#imx = list(np.asarray( img, dtype="float32" ).reshape([224*224*3]))

#img = keras.preprocessing.image.array_to_img(imx)
imgplot = plt.imshow(img)

# COMMAND ----------

imx = np.asarray(img,  dtype=np.double )
type(imx)

# COMMAND ----------

imx.shape

# COMMAND ----------

imx = np.moveaxis(imx, -1, 0)
imx.shape

# COMMAND ----------

type(imx[1][2][2])
imx.dtype

# COMMAND ----------

#from distrib_resnet.load_model import Resnet50Model
import torch
from torchvision import  models

model_state = models.resnet50(pretrained=True).state_dict()
bc_model_state = sc.broadcast(model_state)

# COMMAND ----------

from torchvision import transforms
#
# Create a preprocessing pipeline
#
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
#
# Pass the image for preprocessing and the image preprocessed
#
img_preprocessed = preprocess(img)
#
# Reshape, crop, and normalize the input tensor for feeding into network for evaluation
#
batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)


# COMMAND ----------

model = models.resnet50(pretrained=True)
model.load_state_dict(bc_model_state.value)
m = model.eval()

# COMMAND ----------

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


tgx = preprocess(img)
tgxx = torch.unsqueeze(tgx, 0)
out = model(tgxx)

# COMMAND ----------

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
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

# COMMAND ----------

#out = model(batch_img_tensor)
#tx = torch.tensor(imx,dtype=torch.float)
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tx1 = preprocess(imx)
txx = torch.unsqueeze(tx, 0)#.double()


# COMMAND ----------


