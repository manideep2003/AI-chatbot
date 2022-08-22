import random
import pickle
import json
from asyncore import read
import intents
import numpy as np
import document as document

import nltk
import words as words
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json'),read(intents))

word=[]
classes=[]
documents=[]
ignore_letters=['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        document.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

"""words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))"""


