from torch.utils.data import Dataset


class ImageDataset(Dataset):
  def __init__(self, input_arr, transform=None):
    self.data = np.array(input_arr)
    self.transform = transform
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index):
    x = self.data[index]
    if self.transform is not None:
      x = self.transform(x)
    return x
  
  @staticmethod
  def get_batch_size():
      return 500