#Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os as os
import time
import sys
from dataPrep import fetch_data


device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
device = torch.device("cpu")

batch_size = 64

train_dataset, test_dataset, val_dataset, seqLength, vocabSize, vocab = fetch_data()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocabSize, seqLength)
        self.rnn = nn.GRU(input_size = seqLength, hidden_size = 100, num_layers = 3, batch_first = True)
        self.fc = nn.Linear(in_features= 100, out_features= 6)

    def forward(self, x):
        x = x.to(device)
        x = self.embedding(x)
        x = x.view(-1, seqLength, seqLength) # Reshape input tensor to match (batch_size, sequence_length, input_size)
        h0 = torch.zeros(3, x.size(0),100).to(device) # Initializes the initial hidden state with (layer_dim, batch_size, hidden_dim)
        out, _ = self.rnn(x, h0) # Applies the RNN layers to the input tensor x with the initial hidden state h0.
        hidden_state_outputs = out[:, -1, :] # Returns the hidden state outputs at the last time step of each sequence in the batch.
        result=self.fc(hidden_state_outputs) # This operation transforms the hidden state outputs into the final predictions.
        return result

def train():
    model = RNNModel()
    model = model.to(device)
    print(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Train network
    num_epochs = 10

    train_losses = []
    val_losses = []
    accuracies = []

    modelPB = model
    valPB = -1
    accPB = 50

    # plt.ion()
    # fig, ax1 = plt.subplots()

    # ax2 = ax1.twinx()
    # ax1.plot(train_losses, 'b', label='Training loss')
    # ax1.plot(val_losses, 'r', label='Validation loss')
    # ax2.plot(accuracies, 'g', label='Accuracy')

    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss')
    # ax2.set_ylabel('Accuracy')
    # ax1.set_xlim(0, num_epochs)
    # ax1.set_ylim(0, 5)
    # ax2.set_ylim(20, 100)

    # plt.title('Loss and accuracy')
    # plt.show()

    for epoch in range(num_epochs):
        model.train()
        trainLoss = 0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()

        model.eval()
        correct = 0
        total = 0
        valLoss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(torch.softmax(outputs.detach(), dim=1), dim=1)
                # print(f'Predicted: {predicted}')
                # print(f'Target:    {targets}')
                # print(f'Outputs: {outputs}')
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                valLoss += criterion(outputs, targets).item()

        accuracy = 100 * correct / total
        print(f'Epoch: {epoch}, accuracy: {accuracy}, Train loss: {trainLoss / len(train_loader)}, Validation loss: {valLoss / len(test_loader)}')
        print(f'Correct: {correct}, Total: {total}')
        valLoss = valLoss / len(test_loader)
        trainLoss = trainLoss / len(train_loader)
        train_losses.append(trainLoss)
        val_losses.append(valLoss)
        accuracies.append(accuracy)
        if valPB == -1:
            valPB = valLoss
        if valPB > valLoss:
            valPB = valLoss
            accPB = accuracy
            modelPB = model
            print(valPB)

        # ax1.plot(train_losses)
        # ax1.plot(val_losses)
        # ax2.plot(accuracies)
        # fig.canvas.draw()
        # fig.canvas.flush_events()

    # plt.show(block=True)

    torch.save(modelPB.state_dict(), './model')