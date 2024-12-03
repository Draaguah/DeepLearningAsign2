from emotion import RNNModel
from emotion import train
from transformer import TransformerClassifier
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from dataPrep import fetch_data
import matplotlib.pyplot as plt
import sys
import pickle

# device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
device = torch.device('cpu')


batch_size = 64

train_dataset, test_dataset, val_dataset, seqLength, vocabSize = fetch_data()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

vocab = {}
with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)


def load_model():
    print(f"device: {device}")
    model = TransformerClassifier(vocabSize, 6, d_model=seqLength, d_key=8, n_heads=4, mlp_factor=100, n_layers=4)
    model.load_state_dict(torch.load('DeepLearningAsign2/modelTrans', map_location='cpu', weights_only=True))
    model.eval()
    data, labels = next(iter(test_loader))

    data = data.to(device)
    labels = labels.to(device)

    outputs = model(data).to('cpu')
    _, predicted = torch.max(torch.softmax(outputs, dim=1), dim=1)

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