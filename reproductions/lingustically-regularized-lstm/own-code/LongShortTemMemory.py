import theano

import theano.tensor as T
import numpy as np

from theano import function, shared
from theano.tensor.shared_randomstreams import RandomStreams

class LongShortTermMemory:
    def __init__(self):
        self.srng = RandomStreams(0)

    def init_cell(self):
        def shared_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX)
            f = function([], matrix)
            return shared(f(), name=name)

        h_z, n_l = self.hidden_size, self.num_layers

        u = lambda x : 1 / np.sqrt(x)

        # forget gate
        self.W_f = shared_matrix((h_z, n_l), 'W_f', u(h_z))
        self.U_f = shared_matrix((n_l, n_l), 'U_f', u(n_l))
        self.b_f = shared_matrix((n_l, ), 'b_f', 0.)

        # input gate
        self.W_i = shared_matrix((h_z, n_l), 'W_i', u(h_z))
        self.U_i = shared_matrix((n_l, n_l), 'U_i', u(n_l))
        self.b_i = shared_matrix((n_l, ), 'b_i', 0.)

        # cell gate
        self.W_c = shared_matrix((h_z, n_l), 'W_c', u(h_z))
        self.U_c = shared_matrix((n_l, n_l), 'U_c', u(n_l))
        self.b_c = shared_matrix((n_l, ), 'b_c', 0.)

        # output gate
        self.W_o = shared_matrix((h_z, n_l), 'W_o', u(h_z))
        self.U_o = shared_matrix((n_l, n_l), 'U_o', u(n_l))
        self.b_o = shared_matrix((n_l, ), 'b_o', 0.)

    def forward_cell(self, x_t, h_prev, c_prev):
        # calculate output of forget gate
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + T.dot(h_prev, self.U_f) + self.b_f)

        # calculate output of input gate
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + T.dot(h_prev, self.U_i) + self.b_i)

        # calculate output of internal temporary cell state
        c_internal_t = i_t * T.nnet.tanh(T.dot(x_t, self.W_c) + T.dot(x_t, self.U_c) + self.b_c)

        # calculate output of current cell state
        c_t = c_prev * f_t + c_internal_t

        # calculate output of output gate
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + T.dot(x_t, self.U_o) + self.b_o)

        # calculate output of current hidden state
        h_t = T.nnet.tanh(c_t) * o_t

        # return hidden state and cell state for the next recursion
        return h_t, c_t

