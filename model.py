import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


class ConvPoolLayer(object):
	"""Pool Layer of a convolutional network """
	def __init__(self, rng, input, filter_shape, feature_image_shape, poolsize=(1, 2)):
		assert feature_image_shape[1] == filter_shape[1]
		self.input = input
		fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
		# initialize weights with random weights
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
			dtype=theano.config.floatX),
			borrow=True
		)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# convolve input feature maps with filters
		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			input_shape=feature_image_shape
		)

		# pool each feature map individually, using maxpooling
		pooled_out = pool.pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.params = [self.W, self.b]
		self.input = input


class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		self.input = input
		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
		if activation == theano.tensor.nnet.sigmoid:
			W_values *= 4

		W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		# parameters of the model
		self.params = [self.W, self.b]


class NetworkPreserve(object):
	def __init__(self, input, edges, neg_edges):
		# S = T.dot(input, input.T)
		# n = edges.shape[0]
		# # N = S.norm(2, axis=1)
		# N = [0] * n
		# for i in range(n):
		# 	N[i] = T.sqrt(T.sum(input[i] ** 2))
		# N2 = np.zeros((n, n), dtype=float)
		# for i in range(n):
		# 	for j in range(n):
		# 		N2 = 1./ (N[i] * N[j])
		# S2 = S * N2
		# S2 = distance.cdist(input, input, 'cosine')
		# N = np.linalg.norm(input, axis=1)
		# self.loss = T.mean((edges - S2) ** 2)
		# self.loss = T.mean((edges - T.dot(input, input.T)) ** 2)
		# self.loss = T.mean((edges - 1./(1 + T.exp(-T.dot(input, input.T)))) ** 2)
		self.loss = -T.mean(T.log(1. / (1 + T.exp(-T.batched_dot(input[edges[0]], input[edges[1]]))))) - \
					5 * T.mean(T.log(1. / (1 + T.exp(T.batched_dot(input[neg_edges[0]], input[neg_edges[1]])))))

	def first_order_proximity(self):
		return self.loss


def build_cnn(X=None, edges=None, neg_edges=None, alpha=0.1, n_epochs=20, nkerns=[10, 20], kerns_size=[101, 25], dimension=100):
	rng = np.random.RandomState(23455)
	batch_size = X.shape[0]
	n_channels = X.shape[1]
	n_dimension = X.shape[3]
	sec_conv_input_size = (n_dimension - kerns_size[0] + 1) / 2
	sec_conv_output_size = (sec_conv_input_size - kerns_size[1] + 1) / 2
	hidden_input_size = nkerns[1] * sec_conv_output_size
	hidden_output_size = dimension

	x = T.dmatrix('x')
	layer0_input = x.reshape((batch_size, n_channels, 1, n_dimension))

	print('... building the model')
	# construct the first Convolution-Pooling Layer
	layer0 = ConvPoolLayer(
		rng,
		input=layer0_input,
		feature_image_shape=(batch_size, n_channels, 1, n_dimension),
		filter_shape=(nkerns[0], n_channels, 1, kerns_size[0]),
		poolsize=(1, 2)
	)
	# construct the second Convolution-Pooling Layer
	layer1 = ConvPoolLayer(
		rng,
		input=layer0.output,
		feature_image_shape=(batch_size, nkerns[0], 1, sec_conv_input_size),
		filter_shape=(nkerns[1], nkerns[0], 1, kerns_size[1]),
		poolsize=(1, 2)
	)
	# flatten output of second Convolution-Pooling Layer to 2D features
	layer2_input = layer1.output.flatten(2)

	# construct a fully-connected layer
	layer2 = HiddenLayer(
		rng,
		input=layer2_input,
		n_in=hidden_input_size,
		n_out=hidden_output_size,
		activation=T.tanh
	)

	# compute loss of preserve Network property of first-order proximity
	layer3 = NetworkPreserve(input=layer2.output, edges=edges, neg_edges=neg_edges)
	cost = layer3.first_order_proximity()

	# create a list of all model parameters to be fit by gradient descent
	params = layer2.params + layer1.params + layer0.params
	grads = T.grad(cost, params)

	updates = [
		(param_i, param_i - alpha * grad_i)
		for param_i, grad_i in zip(params, grads)
	]

	train_model = theano.function(
		[layer0_input], [cost, layer2.W, layer2.output],
		updates=updates
	)

	print('... training')
	# training model and print progress
	embeddings = np.zeros((batch_size, dimension), dtype=float)
	prev = None
	for epoch in range(n_epochs):
		cost, params, embeddings = train_model(X)
		product = embeddings.dot(embeddings.T)
		prev = params
		if epoch % 2 == 0:
			print 'Iter:%d, cost:%.8f' % (epoch, cost)
	return embeddings