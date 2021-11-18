import pytest

from distrib_resnet.load_model import load_model, broadcast_model


def test_broadcast_model(spark):
    model = load_model()
    state = broadcast_model(model,spark.sparkContext)
    assert state is not None