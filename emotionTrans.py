import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os as os
import time
from dataPrep import fetch_data
from transformer import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

batch_size = 32

train_dataset, test_dataset, val_dataset, seqLength, vocabSize = fetch_data()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = TransformerClassifier(vocabSize, 6, d_model=seqLength, d_key=8, n_heads=4, mlp_factor=100, n_layers=4)
model = model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train network
num_epochs = 100

train_losses = []
val_losses = []
accuracies = []

modelPB = model
valPB = -1
accPB = 50

plt.ion()
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(train_losses, 'b', label='Training loss')
ax1.plot(val_losses, 'r', label='Validation loss')
ax2.plot(accuracies, 'g', label='Accuracy')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax1.set_xlim(0, num_epochs)
ax1.set_ylim(0, 5)
ax2.set_ylim(20, 100)

plt.title('Loss and accuracy')
plt.show()

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
            print(f'Predicted: {predicted}')
            print(f'Target:    {targets}')
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

    ax1.plot(train_losses)
    ax1.plot(val_losses)
    ax2.plot(accuracies)
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.show(block=True)

torch.save(modelPB.state_dict(), f"DeepLearningAsign2/Outputs/model_{accPB}%_{time.strftime('%Y%m%d-%H%M%S')}")
