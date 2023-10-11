#
# Defining of a neural network and training it, in the end checking the f-1 score
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torcheval.metrics.functional import binary_f1_score
import pickle

# Define neural network class 'Net' for binary classification
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers for the neural network
        self.fc1 = nn.Linear(1536, 400, dtype=torch.float64)
        self.fc2 = nn.Linear(400, 150, dtype=torch.float64)
        self.fc3 = nn.Linear(150, 64, dtype=torch.float64)
        self.fc4 = nn.Linear(64, 1, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

# Define an upgraded neural network class 'UpgradedNet' with BatchNormalization, dropout
class UpgradedNet(nn.Module):
    def __init__(self):
        super(UpgradedNet, self).__init__()
        # Define layers, BatchNormalization, and dropout
        self.fc1 = nn.Linear(1536, 400, dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(400, dtype=torch.float64)
        self.fc2 = nn.Linear(400, 150, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(150, dtype=torch.float64)
        self.fc3 = nn.Linear(150, 64, dtype=torch.float64)
        self.bn3 = nn.BatchNorm1d(64, dtype=torch.float64)
        self.fc4 = nn.Linear(64, 1, dtype=torch.float64)
        self.dropout_less = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply dropout and BatchNormalization in the forward pass
        x = self.dropout_less(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

# Define a Siamese neural network class 'SiameseNetwork' for binary classification
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Define layers, BatchNormalization, dropout, and sigmoid activation
        self.brothers = nn.Sequential(
            nn.Linear(768, 300, dtype=torch.float64),
            nn.ReLU(),
            nn.BatchNorm1d(300, dtype=torch.float64),
            nn.Linear(300, 250, dtype=torch.float64),
            nn.ReLU(),
            nn.BatchNorm1d(250, dtype=torch.float64),
            nn.Linear(250, 200, dtype=torch.float64),
            nn.ReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64)
        )
        self.fc1 = nn.Linear(400, 320, dtype=torch.float64)
        self.fc2 = nn.Linear(320, 64, dtype=torch.float64)
        self.fc3 = nn.Linear(64, 1, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward_once(self, output):
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = F.relu(self.fc2(output))
        output = self.sigmoid(self.fc3(output))
        return output

    def forward(self, input1, input2):
        # Forward common pass for Siamese network
        output1 = self.brothers(input1)
        output2 = self.brothers(input2)
        output = torch.cat((output1, output2), 1)
        output = self.forward_once(output)
        return output

# Create an instance of the neural network
net = SiameseNetwork()  #

# Set batch size for data loading
batch_size = 64

# Load training and test datasets
trainset = pickle.load(open('testset_trainset/trainset.obj', 'rb'))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                          num_workers=2)
# Define loss function
criterion = nn.BCELoss()

# Initialize the optimizer with the initial learning rate
lr = [0.005, 0.001, 0.0005, 0.0001]
k = -1
optimizer = optim.Adam(net.parameters(), lr=lr[0])
massimo = 0

# Training loop
for epoch in range(40):
    if epoch % 10 == 0:
        k += 1
        optimizer = optim.Adam(net.parameters(), lr=lr[k])
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # Get inputs and labels
        inputs, labels = data
        inputs1, inputs2 = torch.split(inputs, 768, dim=1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass and backward pass
        # outputs = net(inputs)
        outputs = net(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    # Calculate and print F1 scores on the test set
    # a, b = torch.split(testset.tensors[0], 768, dim=1)
    # outputs = net(a,b)
    # outputs = net(testset.tensors[0])
    # f1_score_running_test_05 = binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0]
    #                                            .type(torch.int64), threshold=0.5)
    # f1_score_running_test_04 = binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0]
    #                                            .type(torch.int64), threshold=0.4)
    # f1_score_running_test_07 = binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0]
    #                                            .type(torch.int64), threshold=0.7)
    # print(f'[{epoch + 1} f_1 score test set 0.5: {f1_score_running_test_05:.3f}')
    # print(f'[{epoch + 1} f_1 score test set 0.4: {f1_score_running_test_04:.3f}')
    # print(f'[{epoch + 1} f_1 score test set 0.7: {f1_score_running_test_07:.3f}')

    # Save the model with the highest F1 score on the test set
    #if f1_score_running_test_05 > massimo:
     #   massimo = f1_score_running_test_05
     #   print('massimo f1_score: ', f1_score_running_test_05)

    # Calculate and print F1 score on the training set
    # a, b = torch.split(trainset.tensors[0], 768, dim=1)
    # outputs = net(a,b)
    # outputs = net(trainset.tensors[0])
    # f1_score_running_train = binary_f1_score(outputs.t()[0], trainset.tensors[1].t()[0]
    #                                          .type(torch.int64), threshold=0.5)
    # print(f'[{epoch + 1} f_1 score train set: {f1_score_running_train:.3f}')

print('Finished Training')
del trainset
del trainloader
testset = pickle.load(open('testset.obj', 'rb'))
a, b = torch.split(testset.tensors[0], 768, dim=1)
outputs = net(a,b)
# outputs = net(testset.tensors[0])
f1_score_running_test_05 = binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0]
                                               .type(torch.int64), threshold=0.5)
f1_score_running_test_04 = binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0]
                                               .type(torch.int64), threshold=0.4)
f1_score_running_test_07 = binary_f1_score(outputs.t()[0], testset.tensors[1].t()[0]
                                                .type(torch.int64), threshold=0.7)
del testset
torch.save(net.state_dict(), 'models/model_siamese.pth')
