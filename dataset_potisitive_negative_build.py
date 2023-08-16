import pickle
import torch
from torch.utils.data import TensorDataset

f_x = open('x_tensors.obj', 'rb')
x = pickle.load(f_x)
f_x.close()

f_y = open('y_label.obj', 'rb')
y = pickle.load(f_y)
f_y.close()

f_list = open('lista_x_y.obj', 'rb')
lista = pickle.load(f_list)
f_list.close()

for i in range(len(x)):
    x[i] = x[i].t()
    x[i] = x[i][None, :]

x_data = torch.cat((x))

y = torch.tensor(y)
y = y[:,None]

mat = TensorDataset(x_data, y)
