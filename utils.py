from __future__ import print_function
from __future__ import division
import warnings
warnings.simplefilter(action='ignore')
import json, os, argparse
import string
import numpy as np
from tqdm import tqdm
import h5py
import urllib.request as urllib
from keras.utils.data_utils import get_file
from os import path

context_maxlen = 300
ques_maxlen = 25
answer_maxlen = 30

def tokenize(vector, word_dict):
    global i
    from nltk.tokenize import RegexpTokenizer
#     tokenizer = RegexpTokenizer('\w+|\$[\d]+|\S+')
#     tokenizer = RegexpTokenizer('\w+') 
    tokenizer = RegexpTokenizer('(\w+([\'-]\w+)*)')
    tokens = tokenizer.tokenize(vector)
    words = []
    for w in tokens:
        if w[0].isalnum():
            word = w[0].lower()
            if word in word_dict:
                words.append(word_dict[word])
    return words

def find(mylist, sublist):
    h = len(mylist)
    n = len(sublist)
    skip = {sublist[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if mylist[i - j] != sublist[-j - 1]:
                i += skip.get(mylist[i], n)
                break
        else:
            return i - n + 1
    return -1
def pad_sequence(lst, maxlen):
    array = np.zeros((len(lst), maxlen), dtype=np.int)
    for i in range(len(lst)):
        for j in range(min(len(lst[i]) - 1, maxlen)):
            array[i][j] = lst[i][j]
    return array

def padding(context, question, answer, begin, end):
    context_array = pad_sequence(context, context_maxlen)
    question_array = pad_sequence(question, ques_maxlen)
    answer_array = pad_sequence(answer, answer_maxlen)
    begin_array = np.zeros((len(begin), context_maxlen), dtype=np.int)
    end_array = np.zeros((len(end), context_maxlen), dtype=np.int)

    for i in range(len(begin)):
        begin_array[i][min(begin[i], context_maxlen - 1)] = 1 

    for i in range(len(end)):
        end_array[i][min(end[i], context_maxlen - 1)] = 1

    print('context: ', context_array.shape)
    print('question: ', question_array.shape)
    print('answer: ', answer_array.shape)
    print('answer start: ', begin_array.shape)
    print('answer end: ',end_array.shape)
    return context_array, question_array, answer_array, begin_array, end_array
def predict_answer(p):
    start, end = 0, 10
    max_prob = 0.0
    n = 300
    for i in range(n):
        for j in range(i + 2, min(i + 20, n)):
            curr_prob = p[0][0][i] * p[1][0][j]
            if( curr_prob > max_prob):
                start, end, max_prob = i, j, curr_prob
    return start, end
