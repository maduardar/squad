import warnings
warnings.simplefilter(action='ignore')
import string
import numpy as np
import tqdm
from tqdm import tqdm_notebook as tqdm
import h5py
from utils import tokenize, find, pad_sequence, padding, predict_answer

word_dict = np.load('Prepared data/word2ind.npy').item()
ind_dict = np.load('Prepared data/ind2word.npy').item()
print('Enter the context:')
text = str(input())

print('Enter the question:')
ques = str(input())


#context
text = tokenize(text, word_dict)
ques = tokenize(ques, word_dict)
context = np.zeros((1, 300), dtype=np.int)
for j in range(min(len(text), 300)):
    context[0][j] = text[j]
question = np.zeros((1, 25), dtype=np.int)
for i in range(min(len(ques), 25)):
    question[0][i] = ques[i]


import model
from model import RNet, custom_objects
from keras.models import load_model

model = load_model('Weights/new_model.h5', custom_objects=custom_objects())

print('Answer:')

p = model.predict([context, question])
b, e = predict_answer(p)
for i in range(b, e):
    print(ind_dict[context[0][i]], end=' ')
