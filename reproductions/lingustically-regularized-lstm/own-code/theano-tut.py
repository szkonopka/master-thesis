from theano import function
from theano import shared
import theano.tensor as T

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=(state, state + inc))
accumulator(1)