#Imports
import torch
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, TensorDataset
import os as os
import math
import numpy as np


def fetch_data():
    nltk.download('stopwords')
    ds = load_dataset("dair-ai/emotion", "split")
    # print("Train set:")
    # for i in range(6):
    #     count = ds['train']['label'].count(i)
    #     length = len(ds['train']['label'])
    #     print(f'Class: {i}: {count}, percentage of set: {count/length * 100}')

    # print("Test set:")    
    # for i in range(6):
    #     count = ds['test']['label'].count(i)
    #     length = len(ds['test']['label'])
    #     print(f'Class: {i}: {ds['test']['label'].count(i)}, percentage of set: {count/length * 100}')

    # print("Validation set:")
    # for i in range(6):
    #     count = ds['validation']['label'].count(i)
    #     length = len(ds['validation']['label'])
    #     print(f'Class: {i}: {ds['validation']['label'].count(i)}, percentage of set: {count/length * 100}')

    excludedWords = stopwords.words('english')
    trainTokens = [[word for word in tweet.split() if word not in excludedWords] for tweet in ds['train']['text']]
    trainTargets = ds['train']['label']
    testTokens = [[word for word in tweet.split() if word not in excludedWords] for tweet in ds['test']['text']]
    testTargets = ds['test']['label']
    valTokens = [[word for word in tweet.split() if word not in excludedWords] for tweet in ds['validation']['text']]
    valTargets = ds['validation']['label']

    lengths = [len(tweet) for tweet in trainTokens]
    mean = np.mean(lengths)
    std = np.std(lengths)

    print(f'Mean: {mean}')
    print(f'Std: {std}')

    words = set(sum(trainTokens, []))
    vocab = {word: i+1 for i, word in enumerate(words)}
    vocab['$'] = 0

    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    for i, tweet in enumerate(trainTokens):
        print(f'{emotions[trainTargets[i]]}: {tweet}')

    seqLength = int(math.floor(mean + std))
    vocabSize = len(vocab)
    print(seqLength)
    print(vocabSize)

    trainTokens = [(tweet + ['$'] * (seqLength - len(tweet)))[:seqLength] for tweet in trainTokens]
    testTokens = [(tweet + ['$'] * (seqLength - len(tweet)))[:seqLength] for tweet in testTokens]
    valTokens = [(tweet + ['$'] * (seqLength - len(tweet)))[:seqLength] for tweet in valTokens]

    trainEncoded = [[vocab.get(word, 0) for word in tweet] for tweet in trainTokens]
    testEncoded = [[vocab.get(word, 0) for word in tweet] for tweet in testTokens]
    valEncoded = [[vocab.get(word, 0) for word in tweet] for tweet in valTokens]

    train_dataset = TensorDataset(torch.tensor(trainEncoded), torch.tensor(trainTargets))
    test_dataset = TensorDataset(torch.tensor(testEncoded), torch.tensor(testTargets))
    val_dataset = TensorDataset(torch.tensor(valEncoded), torch.tensor(valTargets))

    return (train_dataset, test_dataset, val_dataset, seqLength, vocabSize)