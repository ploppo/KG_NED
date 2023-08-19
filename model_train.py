import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import TensorDataset
import torch.optim as optim
from torcheval.metrics.functional import binary_f1_score

from dataset_potisitive_negative_build import x_data, y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1536, 150, dtype=torch.float64)
        self.fc2 = nn.Linear(150, 84, dtype=torch.float64)
        self.fc3 = nn.Linear(84, 1, dtype=float)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

net = Net()

batch_size = 4

trainset = TensorDataset(x_data[:60000], y[:60000])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset =  TensorDataset(x_data[60000:], y[60000:])
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    f1_score_running = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # f1_score_running += binary_f1_score(outputs.t()[0].type(torch.int64), labels.t()[0].type(torch.int64), threshold=0.4)

        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000 :.3f}') #f1_score :{f1_score_running /2000 :.3f}')
            running_loss = 0.0
            #f1_score_running = 0.0
    outputs = net(testset.tensors[0])
    f1_score_running += binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0].type(torch.int64), threshold=0.5)
    print(f'[{epoch + 1} f_1 score: {f1_score_running:.3f}')

print('Finished Training')