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
from utils import tokenize, find, pad_sequence, padding
import urllib.request as urllib

def get_glove_file_path():
    SERVER = 'http://nlp.stanford.edu/data/'
    VERSION = 'glove.840B.300d'

    origin = '{server}{version}.zip'.format(server=SERVER, version=VERSION)
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'Data')

    fname = '/tmp/glove.zip'
    get_file(fname,
             origin=origin,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    # Remove unnecessary .zip file and keep only extracted .txt version
    os.remove(fname)
    return path.join(cache_dir, VERSION) + '.txt'

url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
urllib.urlretrieve(url+'train-v1.1.json', './Data/train-v1.1.json')
urllib.urlretrieve(url+'dev-v1.1.json', './Data/dev-v1.1.json')
print('Loading train and dev sets...')
with open('./Data/train-v1.1.json') as json_data:
    d = json.load(json_data)
with open('./Data/dev-v1.1.json') as json_data:
    d1 = json.load(json_data)
print('Done!')

url = 'https://www.dropbox.com/s/qtxf5qxabbmfs1v/model.h5?dl=1'
model_path = 'Weights/new_model.h5'

if not path.exists(model_path):
    print('Loading pretrained model...')
    urllib.urlretrieve(url, model_path)
    print('Done!')
else:
    print('Pretrained model already exists!!! (yeah cool)')
    
glove_file_path = 'Data/glove.840B.300d.txt'
if not path.exists(glove_file_path):
    glove_file_path = get_glove_file_path()
print('Data upload completed successfully!')
print('Start preporation...')

emb_size, emb_dim = 2195876, 300
f = open(glove_file_path)
word_dict, ind_dict = {}, {}
embedding_matrix = np.zeros((emb_size, emb_dim))
i = 1
for lines in tqdm(f):
    conv_err = 0
    line = lines.split()
    word = line[0].lower()
    try:
        f = float(line[1])
    except:
        conv_err += 1
    else:
        if word not in word_dict and i < emb_size:
            vec = np.array(line[1:], dtype='float32')
            if vec.shape[0] != emb_dim:
                continue
            word_dict[word] = i
            ind_dict[i] = word
            embedding_matrix[i] = vec
            i += 1 
emb_size = len(word_dict) + 1
embedding_matrix = np.resize(embedding_matrix, (emb_size, emb_dim))
with h5py.File('Prepared data/embeddings.h5', 'w') as hf:
        hf.create_dataset('embeddings', data=embedding_matrix)



def parse_data(dataset):
    context_list = []
    question_list = []
    answer_list = []
    answer_begin = []
    answer_end = []
    error = 0
    
    for article in tqdm(dataset):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                for ans in qa['answers']:
                    # append both context and questions many times for more than one question/answer
                    ques = tokenize(qa['question'], word_dict)
                    if len(ques) < 100:
                        cont = tokenize(paragraph['context'], word_dict)
                        b, e = ans['answer_start'], ans['answer_start']+len(ans['text'])
                        an = tokenize(paragraph['context'][b:e], word_dict)
                        begin = find(cont, an)
                        if begin > 0:
                            question_list.append(ques)
                            context_list.append(cont)
                            answer_list.append(an)
                            answer_begin.append(begin)
                            answer_end.append(begin + len(an))
    return context_list, question_list, answer_list, answer_begin, answer_end


print('Preparing train dataset...')
context_list, question_list, answer_list, answer_begin, answer_end = parse_data(d['data'])
print('Done!')

print('Preparing dev dataset...')
dev_context_list, dev_question_list, dev_answer_list, dev_answer_begin, dev_answer_end = parse_data(d1['data'])
print('Done!')

context_maxlen = 300
ques_maxlen = 25
answer_maxlen = 30

def save_data(data_type):
    print('Shapes of ' + data_type + ' data')
    if data_type == 'test':
        context_array, question_array, answer_array, begin_array, end_array =         padding(context_list, question_list, answer_list, answer_begin, answer_end)
    if data_type == 'dev':
        context_array, question_array, answer_array,begin_array, end_array =         padding(dev_context_list, dev_question_list,
                dev_answer_list, dev_answer_begin, dev_answer_end)
    dr = 'Prepared data/' + data_type + '_'
    with h5py.File(dr + 'context.h5', 'w') as hf:
        hf.create_dataset('context', data=context_array)
    with h5py.File(dr + 'questions.h5', 'w') as hf:
        hf.create_dataset('questions', data=question_array)
    with h5py.File(dr + 'answers.h5', 'w') as hf:
        hf.create_dataset('answers', data=answer_array)
    with h5py.File(dr + 'begin.h5', 'w') as hf:
        hf.create_dataset('begin', data=begin_array)
    with h5py.File(dr + 'end.h5', 'w') as hf:
        hf.create_dataset('end', data=end_array)


save_data('test')
save_data('dev')

dr = 'Prepared data/'
np.save(dr + 'word2ind.npy', word_dict)
np.save(dr + 'ind2word.npy', ind_dict)

print('Success!')
