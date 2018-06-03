import warnings
warnings.simplefilter(action='ignore')
import string
import numpy as np
import tqdm
from tqdm import tqdm_notebook as tqdm
import h5py
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras as keras
from keras import backend as K

from theano import ifelse

model_path = 'Weights/new_model.h5'

path = 'Prepared data/'
def h5load(file_name, emb=False):
    pth = path
    if not emb:
        pth += 'dev_'
    with h5py.File(pth + file_name + '.h5', 'r') as hf:
        return hf[file_name][:]
context = h5load('context')
context_maxlen = context.shape[1]
answers = h5load('answers')
answer_maxlen = answers.shape[1]
questions = h5load('questions')
ques_maxlen = questions.shape[1]
begin = h5load('begin')
end = h5load('end')
embedding_matrix = h5load('embeddings', True)
word2ind = np.load(path + 'word2ind.npy').item()
ind2word = np.load(path + 'ind2word.npy').item()

n = context.shape[0]
emb_dim = 300
vocab_size = len(word2ind)


import model
from model import RNet, custom_objects
from keras.models import load_model
print('Loading pretrained model...')
model = load_model(model_path, custom_objects=custom_objects())
print('Done!')
X, y_true = [context[:], questions[:]], [begin[:], end[:]]

p = model.predict(X, verbose=1)

def predict_answer(p):
    start, end = 0, 10
    max_prob = 0.0
    n = p[0].shape[0]
    for i in range(n):
        for j in range(i, min(i + 20, n)):
            curr_prob = p[0][i] * p[1][j]
            if( curr_prob > max_prob):
                start, end, max_prob = i, j, curr_prob
    return start, end
def TP(pred, true):
    k = 0
    for i in range(pred[0], pred[1] + 1):
        if i >= true[0] and i <= true[1]:
            k += 1
    return k

def F1_score(true, pred):
    n = int(pred[0].shape[0])
    print(n)
    f1 = 0.0
    k = 0
    for i in tqdm(range(n)):
        start_true, end_true = np.argmax(true[0][i]),  np.argmax(true[1][i])
        if start_true == end_true:
            k += 1
            continue
        start_pred, end_pred = predict_answer([pred[0][i], pred[1][i]])
        tp = TP([start_pred, end_pred], [start_true, end_true])
        precision = tp / (end_pred - start_pred + 1)
        recall = tp / (end_true - start_true + 1)
        if precision + recall != 0:
            f1 += 2 * precision * recall / (precision + recall)
    return f1 / (n - k)

f1 = F1_score(y_true, p)

print('F1-score = ', f1)

