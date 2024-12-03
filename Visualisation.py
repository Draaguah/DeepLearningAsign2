from emotion import RNNModel
from emotion import train
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from dataPrep import fetch_data
import matplotlib.pyplot as plt
import sys


batch_size = 64

train_dataset, test_dataset, val_dataset, seqLength, vocabSize, vocab = fetch_data()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def load_model():
    model = RNNModel()
    model.load_state_dict(torch.load('model', map_location=torch.device('cpu')))
    model.eval()
    data, labels = next(iter(val_loader))

    outputs = model(data).to('cpu')
    _, predicted = torch.max(torch.softmax(outputs.detach(), dim=1), dim=1)

    return labels, predicted, data


labels, predicted, tweets = load_model()


inv_vocab = {v: k for k, v in vocab.items()}
emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


for i, tweet in enumerate(tweets):
    predict = emotions[predicted[i]]
    label = emotions[labels[i]]
    sys.stdout.write("\033[0;32m" if predict == label else "\033[1;31m")
    print(f'Prediced: {predict}, True: {label}')
    print([inv_vocab.get(word.item()) for word in tweet])
    print()
    sys.stdout.write("\033[0;0m")






ccd = confusion_matrix(labels, predicted)
#cm = multilabel_confusion_matrix(labels, predicted)
#ConfusionMatrixDisplay(ccd).plot() #CM displayed 

ConfusionMatrixDisplay.from_predictions(labels, predicted)
plt.show()