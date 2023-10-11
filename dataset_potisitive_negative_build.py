import pickle
import torch
from torch.utils.data import TensorDataset

f_x_train = open('x_y/MM/x_tensors_full_train_name.obj', 'rb')
x_train = pickle.load(f_x_train)
f_x_train.close()

f_y_train = open('x_y/BC5CDR/y_label_full_train_name.obj', 'rb')
y_train = pickle.load(f_y_train)
f_y_train.close()

x_train = torch.cat(x_train)
y_train = torch.tensor(y_train, dtype=torch.float64).reshape(-1, 1)

trainset = (TensorDataset(x_train, y_train))

f = open('testset_trainset/trainset_name.obj','wb')
pickle.dump(trainset,f)
f.close()

f_x_test = open('x_y/BC5CDR/x_tensors_full_test_NO_RANDOM_BC5CDR.obj', 'rb')
x_test = pickle.load(f_x_test)
f_x_test.close()

f_y_test = open('x_y/BC5CDR/y_label_full_test_NO_RANDOM_BC5CDR.obj', 'rb')
y_test = pickle.load(f_y_test)
f_y_test.close()

x_test = torch.cat(x_test)
y_test = torch.tensor(y_test, dtype=torch.float64).reshape(-1, 1)

testset =  TensorDataset(x_test, y_test)

f = open('testset_trainset/testset_NO_RANDOM_BC5CDR.obj','wb')
pickle.dump(testset,f)
f.close()