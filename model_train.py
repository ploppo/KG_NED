import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torch.utils.data import TensorDataset
import torch.optim as optim
from torcheval.metrics.functional import binary_f1_score

from dataset_potisitive_negative_build import x_train, y_train, x_test, y_test


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.brothers = torch.nn.Sequential(
            torch.nn.Linear(768, 300, dtype=torch.float64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(300, 250, dtype=torch.float64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(250, 200, dtype=torch.float64),
            torch.nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(400, 320, dtype=torch.float64)
        self.fc2 = nn.Linear(320, 64, dtype=torch.float64)
        self.fc3 = nn.Linear(64, 1, dtype=torch.float64)

        self.sigmoid = nn.Sigmoid()
    def forward_once(self, output):
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.sigmoid(self.fc3(output))
        return output
    def forward(self, input1, input2):
        # get two images' features
        output1 = self.brothers(input1)
        output2 = self.brothers(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.forward_once(output)

        return output
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1536, 250, dtype=torch.float64)
        self.fc2 = nn.Linear(250, 150, dtype=torch.float64)
        self.fc3 = nn.Linear(150, 84, dtype=torch.float64)
        self.fc4 = nn.Linear(84, 1, dtype=float)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

net = Net() # SiameseNetwork() # Net()

batch_size = 5

trainset = TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset =  TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # inputs1, inputs2 = torch.split(inputs,768,dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # outputs = net(inputs1, inputs2)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000 :.3f}') #f1_score :{f1_score_running /2000 :.3f}')
            running_loss = 0.0

    # a, b = torch.split(testset.tensors[0], 768, dim=1)
    f1_score_running_test_05 = 0.0
    f1_score_running_test_04 = 0.0
    f1_score_running_test_07 = 0.0
    outputs = net(testset.tensors[0])
    f1_score_running_test_05 += binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0].type(torch.int64), threshold=0.5)
    f1_score_running_test_04 += binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0].type(torch.int64), threshold=0.4)
    f1_score_running_test_07 += binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0].type(torch.int64), threshold=0.7)
    print(f'[{epoch + 1} f_1 score test set 0.5: {f1_score_running_test_05:.3f}')
    print(f'[{epoch + 1} f_1 score test set 0.4: {f1_score_running_test_04:.3f}')
    print(f'[{epoch + 1} f_1 score test set 0.7: {f1_score_running_test_07:.3f}')
    f1_score_running_train = 0.0
    outputs = net(trainset.tensors[0])
    f1_score_running_train += binary_f1_score(outputs.t()[0], trainset.tensors[1].t()[0].type(torch.int64), threshold=0.5)
    print(f'[{epoch + 1} f_1 score test set: {f1_score_running_train:.3f}')

print('Finished Training')