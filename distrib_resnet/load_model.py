import torch
from torch.utils.data import Dataset
from torchvision import  models

def get_model():
  """
  Define model: pretrained resnet50
  """
  return models.resnet50(pretrained=True)

def load_model():
    """
    Load model on current node 
    """
    model_state = get_model().state_dict()
    return model_state

def broadcast_model(model_state,sc):
  """
  Broadcast models state to all nodes for distributed inference
  """
  bc_model_state = sc.broadcast(model_state)
  return bc_model_state


def get_model_for_eval(bc_model_state):
  """Gets the broadcasted model."""
  model = get_model()
  model.load_state_dict(bc_model_state.value)
  model.eval()
  return model
