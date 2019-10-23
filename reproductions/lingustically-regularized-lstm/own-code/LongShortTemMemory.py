import theano

import theano.tensor as T
import numpy as np

from theano import function, shared
from theano.tensor.shared_randomstreams import RandomStreams

class SemanticRegularizer:
    pass

class LongShortTermMemory:
    def __init__(self):z
        self.srng = RandomStreams(0)
        self.hidden_size = 300
        self.num_layers = 300

    def init_cell(self):
        def random_init_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX)
            f = function([], matrix)
            return shared(f(), name=name)

        h_z, n_l = self.hidden_size, self.num_layers

        u = lambda x : 1 / np.sqrt(x)

        # forget gate weights and bias
        self.W_f = random_init_matrix((h_z, n_l), 'W_f', u(h_z))
        self.U_f = random_init_matrix((n_l, n_l), 'U_f', u(n_l))
        self.b_f = random_init_matrix((n_l, ), 'b_f', 0.)

        # input gate weights and bias
        self.W_i = random_init_matrix((h_z, n_l), 'W_i', u(h_z))
        self.U_i = random_init_matrix((n_l, n_l), 'U_i', u(n_l))
        self.b_i = random_init_matrix((n_l, ), 'b_i', 0.)

        # cell gate weights and bias
        self.W_c = random_init_matrix((h_z, n_l), 'W_c', u(h_z))
        self.U_c = random_init_matrix((n_l, n_l), 'U_c', u(n_l))
        self.b_c = random_init_matrix((n_l, ), 'b_c', 0.)

        # output gate weights and bias
        self.W_o = random_init_matrix((h_z, n_l), 'W_o', u(h_z))
        self.U_o = random_init_matrix((n_l, n_l), 'U_o', u(n_l))
        self.b_o = random_init_matrix((n_l, ), 'b_o', 0.)

    def forward_cell(self, x_t, h_prev, c_prev):
        # calculate output of forget gate - decide which data should be remove
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + T.dot(h_prev, self.U_f) + self.b_f)

        # calculate output of input gate - decide which data should be add
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + T.dot(h_prev, self.U_i) + self.b_i)

        # calculate output of internal temporary cell - implies clear cell state based on previous hidden state nad current input
        c_internal_t = i_t * T.nnet.tanh(T.dot(x_t, self.W_c) + T.dot(x_t, self.U_c) + self.b_c)

        # calculate output of current cell state - with influence of forget, input gates and internal temporary cell
        c_t = c_prev * f_t + c_internal_t

        # calculate output of output gate
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + T.dot(x_t, self.U_o) + self.b_o)

        # calculate output of current hidden state
        h_t = T.nnet.tanh(c_t) * o_t

        # return hidden state and cell state for the next recursion
        return h_t, c_t


