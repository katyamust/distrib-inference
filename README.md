# distrib-inference

This repo demonstrates distributed inference approach of pytorch models using Spark API.

We use pandas_udf functions to provide optimal distribution of Spark Dataframe to cluster nodes. 

# Problem statement

Datasets often comes as numpy matrices input. This sample shows how to inference Dataframe of numpy arrays with pytorch model in distributed way. We create pytorch Dataset and use pretrained model for image classification Resnet50. Images are converted to numpy arrays and loaded into Dataframe.

The repo is work in progress.

