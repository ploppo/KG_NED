import pickle
import torch
from torch.utils.data import TensorDataset

f_x_train = open('x_tensors_full_train.obj', 'rb')
x_train = pickle.load(f_x_train)
f_x_train.close()

f_y_train = open('y_label_full_train.obj', 'rb')
y_train = pickle.load(f_y_train)
f_y_train.close()

x_train = torch.cat((x_train))

y_train = torch.tensor(y_train, dtype=torch.float64).reshape(-1, 1)


f_x_test = open('x_tensors_full_test_dev.obj', 'rb')
x_test = pickle.load(f_x_test)
f_x_test.close()

f_y_test = open('y_label_full_test_dev.obj', 'rb')
y_test = pickle.load(f_y_test)
f_y_test.close()

x_test = torch.cat((x_test))

y_test = torch.tensor(y_test, dtype=torch.float64).reshape(-1, 1)

# mat_train = TensorDataset(x_data_train, y_train)
