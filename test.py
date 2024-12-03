from emotion import RNNModel
from emotion import train
import torch
from dataPrep import fetch_data
import sys
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset
import pickle

train_dataset, test_dataset, val_dataset, seqLength, vocabSize = fetch_data()
nltk.download('stopwords')
ds = load_dataset("dair-ai/emotion", "split")

vocab = {}
with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

excludedWords = stopwords.words('english')
emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

model = RNNModel()
model.load_state_dict(torch.load('DeepLearningAsign2/model', map_location=torch.device('cpu')))
model.eval()

for line in sys.stdin:
    if 'q' == line.rstrip():
        break
    input = [word for word in line.split() if word not in excludedWords]
    tokens = (input + ['$'] * (14 - len(input)))[:14]
    encoded = [vocab.get(word, 0) for word in tokens]
    dataset = torch.tensor(encoded)

    outputs = model(dataset).to('cpu')
    _, predicted = torch.max(torch.softmax(outputs.detach(), dim=1), dim=1)
    print(f'Mood : {emotions[predicted]}')