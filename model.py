# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
warnings.simplefilter(action='ignore')

import keras as keras
from keras import backend as K

from theano import ifelse

import os
os.environ['KERAS_BACKEND'] = 'theano'
THEANO_FLAGS='device=cuda,floatX=float32'


from keras.models import Model, Sequential
from keras.layers import Input, InputLayer
from keras.layers.core import Dense, RepeatVector, Masking, Dropout
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from layers.helpers import compute_mask

from layers import QuestionAttnGRU
from layers import SelfAttnGRU
from layers import PointerGRU
from layers import QuestionPooling
from layers import VariationalDropout
from layers import Slice
from layers import SharedWeight
import h5py
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
K.clear_session()
class RNet(Model):
    def __init__(self, inputs=None, outputs=None,
                       N=None, M=None, C=25, unroll=False,
                       hdim=75, word2vec_dim=300,
                       dropout_rate=0,
                       char_level_embeddings=False,
                       **kwargs):
        # Load model from config
        if inputs is not None and outputs is not None:
            super(RNet, self).__init__(inputs=inputs,
                                       outputs=outputs,
                                       **kwargs)
            return

        '''Dimensions'''
        B = None
        H = hdim
        W = word2vec_dim

        v = SharedWeight(size=(H, 1), name='v')
        WQ_u = SharedWeight(size=(2 * H, H), name='WQ_u')
        WP_u = SharedWeight(size=(2 * H, H), name='WP_u')
        WP_v = SharedWeight(size=(H, H), name='WP_v')
        W_g1 = SharedWeight(size=(4 * H, 4 * H), name='W_g1')
        W_g2 = SharedWeight(size=(2 * H, 2 * H), name='W_g2')
        WP_h = SharedWeight(size=(2 * H, H), name='WP_h')
        Wa_h = SharedWeight(size=(2 * H, H), name='Wa_h')
        WQ_v = SharedWeight(size=(2 * H, H), name='WQ_v')
        WPP_v = SharedWeight(size=(H, H), name='WPP_v')
        VQ_r = SharedWeight(size=(H, H), name='VQ_r')

        shared_weights = [v, WQ_u, WP_u, WP_v, W_g1, W_g2, WP_h, Wa_h, WQ_v, WPP_v, VQ_r]

        P_vecs = Input(shape=(N, ), name='P_vecs')
        Q_vecs = Input(shape=(M, ), name='Q_vecs')

        if char_level_embeddings:
            P_str = Input(shape=(N, C), dtype='int32', name='P_str')
            Q_str = Input(shape=(M, C), dtype='int32', name='Q_str')
            input_placeholders = [P_vecs, P_str, Q_vecs, Q_str]

            char_embedding_layer = TimeDistributed(Sequential([
                InputLayer(input_shape=(C,), dtype='int32'),
                Embedding(input_dim=127, output_dim=H, mask_zero=True),
                Bidirectional(GRU(units=H))
            ]))

            # char_embedding_layer.build(input_shape=(None, None, C))

            P_char_embeddings = char_embedding_layer(P_str)
            Q_char_embeddings = char_embedding_layer(Q_str)

            P = Concatenate() ([P_vecs, P_char_embeddings])
            Q = Concatenate() ([Q_vecs, Q_char_embeddings])

        else:
            P = P_vecs
            Q = Q_vecs
            input_placeholders = [P_vecs, Q_vecs]
        
        with h5py.File('Prepared data/embeddings.h5', 'r') as hf:
            embedding_matrix = hf['embeddings'][:]
        vocab_size, emb_dim = embedding_matrix.shape[0], embedding_matrix.shape[1]
        
        P = Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[embedding_matrix])(P)
        
        uP = Masking() (P)
        for i in range(3):
            uP = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=dropout_rate,
                                   unroll=unroll)) (uP)
        uP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uP') (uP)

        Q = Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[embedding_matrix])(Q)
        uQ = Masking() (Q)
        for i in range(3):
            uQ = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=dropout_rate,
                                   unroll=unroll)) (uQ)
        uQ = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uQ') (uQ)
        uQ_mask = compute_mask(Q)
        vP = QuestionAttnGRU(units=H,
                             return_sequences=True,
                             unroll=unroll) ([
                                 uP, uQ,
                                 WQ_u, WP_v, WP_u, v, W_g1
                             ])
        vP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, H), name='vP') (vP)

        hP = Bidirectional(SelfAttnGRU(units=H,
                                       return_sequences=True,
                                       unroll=unroll)) ([
                                          vP, vP,
                                          WP_v, WPP_v, v, W_g2
                                      ])

        hP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='hP') (hP)

        gP = Bidirectional(GRU(units=H,
                               return_sequences=True,
                               unroll=unroll)) (hP)

        rQ = QuestionPooling() ([uQ, WQ_u, WQ_v, v, VQ_r])
        rQ = Dropout(rate=dropout_rate, name='rQ') (rQ)

        
        fake_input = GlobalMaxPooling1D() (P)
        fake_input = RepeatVector(n=2, name='fake_input') (fake_input)

        
        ps = PointerGRU(units=2 * H,
                        return_sequences=True,
                        initial_state_provided=True,
                        name='ps',
                        unroll=unroll) ([
                            fake_input, gP,
                            WP_h, Wa_h, v,
                            rQ
                        ])

        answer_start = Slice(0, name='answer_start') (ps)
        answer_end = Slice(1, name='answer_end') (ps)

        inputs = input_placeholders + shared_weights
        outputs = [answer_start, answer_end]

        super(RNet, self).__init__(inputs=inputs,
                                   outputs=outputs,
                                   **kwargs)
def custom_objects():
    d = dict()
    from layers import Argmax
    from layers import SharedWeightLayer, SharedWeight
    d['QuestionAttnGRU'] = QuestionAttnGRU
    d['SelfAttnGRU'] = SelfAttnGRU
    d['PointerGRU'] = PointerGRU
    d['QuestionPooling'] = QuestionPooling
    d['Argmax'] = Argmax
    d['Slice'] = Slice
    d['SharedWeightLayer'] = SharedWeightLayer
    d['SharedWeight'] = SharedWeight
    d['VariationalDropout'] = VariationalDropout
    d['RNet'] = RNet
    return d