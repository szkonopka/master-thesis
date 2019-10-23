from theano import function
from theano import shared
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import time
import theano

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state + inc)])
print(accumulator(1))

dimension = 300
srng = RandomStreams(int(1000 * time.time()) % 19921229)
matrix  = srng.uniform((dimension, dimension), low=-1, high=1, dtype=theano.config.floatX) - 100
f = function([], matrix)
print(f())
shared_matrix = shared(f(), name="copied")

print("Test")
print(shared_matrix.get_value())

