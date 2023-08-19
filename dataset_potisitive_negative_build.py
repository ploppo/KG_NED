import pickle
import torch
from torch.utils.data import TensorDataset

f_x = open('x_tensors_small.obj', 'rb')
x = pickle.load(f_x)
f_x.close()

f_y = open('y_label_small.obj', 'rb')
y = pickle.load(f_y)
f_y.close()

x_data = torch.cat((x))

y = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)


mat = TensorDataset(x_data, y)
