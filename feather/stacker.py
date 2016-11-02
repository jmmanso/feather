import numpy as np
import json
import ann
import aux_functions as af





def encode(data, encoder):
	""" Data embedder.
    Transforms a data array to an encoded space.

    Parameters
    ----------
    data: numpy float array of shape (Nsamples, Ndim)

    encoder: ANN trained instance in autoencoder mode.

    Returns
    ----------
    data_encoded: numpy float array with shape
        (Nsamples, Ndim_encoder)
    """
    # Get the list of hidden depths
	hd = encoder.hidden_depths
    # Find the middle hidden layer
	middle_layer_index = (len(hd)-1)/2
    # Initialize empty container for the encoded data
	data_encoded = np.zeros((data.shape[0],hd[middle_layer_index]))
	for i, d_ in enumerate(data):
        # feed forward, get all the activations, and just keep
        # the middle layer, which is the encoding
		x, z_container, x_container = encoder.ff(d_,True,True)
		x_encoded = x_container[1+middle_layer_index]
		data_encoded[i] = x_encoded
	#
	return data_encoded


class Autoencoder:
    """
    Implementation of a deep stacked autoencoder. Uses ann.ANN
    as base class for MLPs.

    The goal is to embed the input data (Nsamples, Ndim) into
    an encoded representation (Nsamples, Ndim_encoded), which is
    useful for dimensionality reduction.

    Given an list of `hidden_depths`, where the last element
    represents the encoding dimension Ndim_encoded, a 1-hidden-layer autoencoder will be fit sequentially with each hidden depth. The
    encoded layer at each step is ported as the input layer of the
    next. After all iterations are performed, all the trained hidden
    layers are stacked together and the full deep autoencoder is formed. This final autoencoder is an ann.ANN instance, which can
    be fit over more epochs in a straighforward manner and thus fine
    tune the network.

    For theoretical insights see:
    https://www.cs.toronto.edu/~hinton/science.pdf

    Parameters
    ----------
    train_X : numpy array of shape = (Nsamples, Ndim)
        The training input data.

    hidden_depths: list of integers
        The sizes of the hidden layers. The last element in this
        list sets the size of the encoded data.

    train_norm_coeff: optional, numpy array of shape = (Nsamples,)
        Normalization weights for the training samples. Affects the
        cost function.

    epochs: int, number of training epochs PER autoencoding step.

    eta: float, learning rate.

    lamb: float, l2 regularization coeffcient.

    batch_size: int, number of samples per mini batch in SGD

    activation_type: string, name of activation function.
        See aux_functions.Activation for a list of options.
    """

    def __init__(self,
        train_X,
        hidden_depths=[20,10,5],
        train_norm_coeff=None,
        epochs=10,
    	eta=0.05,
        lamb=1,
        batch_size=20,
        activation_type='tanh'
        ):

    	self.train_X = train_X
    	self.data_container = [train_X]
    	self.weights_container = []
    	self.bias_container = []
    	self.hidden_depths = hidden_depths
        # To create the full list of encoder-decoder depths,
        # mirror the hidden_depths
    	self.stack_hidden_depths = self.hidden_depths +\
            self.hidden_depths[:-1][::-1]
    	self.epochs = epochs
    	self.eta = eta
    	self.lamb = lamb
    	self.batch_size = batch_size
    	self.activation_type = activation_type
    	self.train_norm_coeff = train_norm_coeff

    def train_miniautoencoder(self, train_X, depth):
        """
        Trains a 1-hidden-layer autoencoder. Appends
        encoded data (hidden activation layer) to a list.

        Parameters
        ----------
        train_X : numpy array of shape = (Nsamples, Ndim)
            The training input data.

        depth: integer, determines the depth of the single
            hidden layer
        """
        # Initialize mininet
        network = ann.ANN(train_X,
                    hidden_depths=[depth],
                    eta=self.eta,
                    lamb=self.lamb,
                    batch_size=self.batch_size,
                    activation_type=self.activation_type)
        # Fit net
        network.fit(self.epochs)
        # Feedforward data array and obtain encoded data array
        data_encoded = encode(train_X, network)
        # Store in class container
        self.data_container.append(data_encoded)
        self.weights_container.append(network.weights)
        self.bias_container.append(network.biases)

    def iterator(self):
        """
        Train mini-autoencoders sequentially.
        """
        for i in range(len(self.hidden_depths)):
        	# Fetch the depth of the current hidden layer
        	depth = self.hidden_depths[i]
        	# Fetch inner data layer
        	data = self.data_container[-1]
        	print 'Starting run %s' % i
        	self.train_miniautoencoder(data, depth)

    def init_stacked_net(self, train_X):
        """
        Initializes the network that will hold the stacked
        autoencoder.

        Parameters
        ----------
        train_X : numpy array of shape = (Nsamples, Ndim)
            The training input data.

        """
        self.stacked_net = ann.ANN(train_X,
                    hidden_depths=self.stack_hidden_depths,
                    eta=self.eta,
                    lamb=self.lamb,
                    batch_size=self.batch_size,
                    activation_type=self.activation_type,
                    train_norm_coeff=self.train_norm_coeff)
        #
        self.stacked_net.tied_weights = False


    def stack(self):
        """
        Stack all the individual and sequential 1-hidden-layer
        networks together.
        """
        # Fetch the zeroth layer data, which is the original input
    	data = self.data_container[0]
        # Initialize network that will contain the stack
    	self.init_stacked_net(data)
        # Add the weights layer by layer from the individual networks.
    	# The weights container has [(I_1,O_1),(I_2,O_2),...(I_n,O_n)],
    	# you need to unfold it as I_1,I_2...I_n:O_n,...O_2,O_1
    	self.stacked_net.weights = [a[0] for a \
            in self.weights_container] + [a[1] for a \
            in self.weights_container][::-1]
    	self.stacked_net.biases = [a[0] for a in self.bias_container]\
         + [a[1] for a in self.bias_container][::-1]

    def run_stacker(self):
        """
        Train mini-autoencoders and stack them together
        """
    	self.iterator()
    	self.stack()


    def save_net(self, file_path):
        """
        Saves autoencoder params to file.
        """
    	serialized_weights = [w.tolist() for w \
        in self.stacked_net.weights]
    	serialized_biases = [b.tolist() for b \
        in self.stacked_net.biases]
    	params = {'weights':serialized_weights,\
         'biases':serialized_biases}
    	with open(file_path,'w') as f:
    		f.write(json.dumps(params))


    def load_net(self, file_path):
        """
        Loads autoencoder params from file.
        """
    	with open(file_path,'r') as f:
    		params = json.loads(f.read())
    	#
    	weights = np.array(params['weights'])
    	biases = np.array(params['biases'])
    	# Since ann.ANN needs to be initialized with some data, which
    	# we dont have yet, we are gonna make a canvas array with
    	# the correct dimensions from the weights
    	fake_data = np.array([np.zeros(len(weights[-1]))])
    	# initialize stacked net
    	self.init_stacked_net(fake_data)
    	# fill in weights and biases
    	self.stacked_net.weights = weights
    	self.stacked_net.biases = biases

















#
