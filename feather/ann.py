import numpy as np
import aux_functions as af

class ANN:
    """
    This class implements a Multilayer Perceptron.


    Parameters
    ----------
    train_X : numpy array of shape = (Nsamples, Nchannels)
        The training input data.

    train_y : optional, numpy array of shape = (Nsamples, Noutput)
        The training target data. If `None`, train_y = train_X

    test_X : optional, numpy array of shape = (Nsamples, Nchannels)
        The test input data. Used only to evaluate loss
        function and compare to training.

    test_y : optional, numpy array of shape = (Nsamples, Noutput)
        The test target data.

    hidden_depths: list of integers
        The sizes of the hidden layers.

    train_norm_coeff: optional, numpy array of shape = (Nsamples,)
        Normalization weights for the training samples. Affects the
        cost function.

    test_norm_coeff: optional, numpy array of shape = (Nsamples,)
        Normalization weights for the test samples. Affects the
        cost function.

    eta: float, learning rate.

    lamb: float, l2 regularization coeffcient.

    batch_size: int, number of samples per mini batch in SGD

    initzero: boolean, whether to initialize weights with zeros

    activation_type: string, name of activation function.
        See aux_functions.Activation for a list of options.

    softmax: boolean, whether to apply softmax in the last layer.
        For example, if activation_type=`relu` and softmax=True,
        only the last layer will be softmax, all others are relu.

    cost_type: string, name cost function. Can be `Xentropy`
        (for classification) or `Gaussian` (for regression)
    """
    def __init__(self,
    	train_X,
    	train_y=None,
    	hidden_depths=[10,10],
    	test_X=None,
    	test_y=None,
    	train_norm_coeff=None,
    	test_norm_coeff=None,
    	eta=1.0,
    	lamb=0.0,
    	batch_size=100,
    	initzero=False,
    	activation_type='relu',
    	softmax=True,
    	cost_type='Xentropy'):

        self.train_X = np.array(train_X)
    	self.Nsamples, self.Nchannels = self.train_X.shape
    	# If test data is not passed and thus None,
    	# evaluation of test data will be skipped
    	self.test_X = test_X
    	self.test_y = test_y
    	# use Softmax for last layer?
    	self.softmax = softmax
    	# activation function name
    	self.activation_type = activation_type
        # cost function name
    	self.cost_type = cost_type

        # If no target samples are given, set up
        # autoencoder
    	if type(train_y).__name__=='NoneType':
    		self.autoencoding = True
    		self.tied_weights = True
    		self.softmax = False
    		self.cost_type = 'Gaussian'
    		self.train_y = self.train_X.copy()
    	else:
    		self.train_y = np.array(train_y)
    		self.autoencoding = False
    		self.tied_weights = False

    	self.activation = af.Activation(self.activation_type)
    	self.last_activation_type = 'softmax' if self.softmax \
        else self.activation_type
    	self.cost_object =\
         af.L_cost(self.cost_type,self.last_activation_type)

    	# NOTE: weight factor renormalization needs to be
    	# done on a batch basis.
    	# normalize weight array so that the sum equals
    	# the array size
    	if type(train_norm_coeff).__name__=='NoneType':
    		self.train_norm_coeff = np.ones(self.train_X.shape[0])
    	else:
    		self.train_norm_coeff = np.array(train_norm_coeff)

    	self.train_norm_coeff =\
        self.train_norm_coeff/sum(self.train_norm_coeff)

    	# if test data exists but not test weights, make test weights all ones
    	if type(test_norm_coeff).__name__=='NoneType'\
         and type(test_X).__name__!='NoneType':
    		self.test_norm_coeff = np.ones(self.test_X.shape[0])
    		self.test_norm_coeff = self.test_norm_coeff/sum(self.test_norm_coeff)
    	# if test data and test weights exits, make sure test weights are array
        elif type(test_norm_coeff).__name__!='NoneType' \
        and type(test_X).__name__!='NoneType':
    		self.test_norm_coeff = np.array(test_norm_coeff)
    		self.test_norm_coeff = self.test_norm_coeff/sum(self.test_norm_coeff)
    	# if test data does not exist but test weights exist, void test weights
        elif type(test_norm_coeff).__name__!='NoneType' \
        and type(test_X).__name__=='NoneType':
    		print 'WARNING: test weights supplied, but not test data. Voiding weights.'
    		self.test_norm_coeff = None
    	else:
    		self.test_norm_coeff = None

    	#
    	self.hidden_depths = np.array(hidden_depths)
    	self.eta = eta
    	self.lamb = lamb
    	self.batch_size = batch_size
    	# Initialize weights and biases as zeroes
    	self.initzero = initzero
        # Determine layer dimensions
    	self.Noutput = self.train_y.shape[1]
    	self.input_depth = [self.Nchannels]
    	self.output_depth = [self.Noutput]
    	self.layer_depths = np.r_[self.input_depth, \
    		self.hidden_depths, self.output_depth]
    	self.Nlayers = len(self.layer_depths)

    	self.Nbatches = self.Nsamples // self.batch_size
    	#
    	# Initialize weights container
    	self.init_weights()
        # Initialize best network as empty tuple
    	self.best_net = ()
        # Containers to track accuracy and cost
    	self.train_cost_container = []
    	self.train_accuracy_container = []
    	self.train_epoch_accuracies = []
    	self.test_epoch_accuracies = []
    	self.sentinel_action_counter = 0


    def init_weights(self):
    	''' Creates containers of arrays for biases and weights.
    	The bias container will have one array per layer (skipping the
    	0th layer), each array of length equal to the number of neurons.
    	The weights container will also have one array per layer, also
    	skipping the 0th one. Each array will be 2-D, with shape of
    	Nneurons(layer) X Nneurons(layer-1).

    	The first and second axis in weights follow the layers l+1,l; like w[k,j].
    	For the bias it is straighforward: b[k]
    	 '''
    	if self.initzero:
    		self.biases = [np.zeros(y) for y in self.layer_depths[1:]]
    		self.weights = [np.zeros((y,x)) for x, y in zip(self.layer_depths[:-1], self.layer_depths[1:])]
    	else:
    		# Use randn(shape) to generate random-normal numbers around mu=0 and std=1
    		self.biases = [np.random.randn(y) for y in self.layer_depths[1:]]
    		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.layer_depths[:-1], self.layer_depths[1:])]

    	if self.tied_weights:
    		# tie weights of second layer to first. Assume there are only 2 total layers
    		self.weights = [self.weights[0],self.weights[0].T]


    def ff(self,x_0, return_layers=False, use_best=False):
    	""" Feed-forward function.
    	Takes a layer=0 input vector, returns layer=L output vector.

        Parameters
        ----------
        x_0 : numpy array of shape = (Nchannels,)
            layer=0 input vector

        return_layers: boolean, whether to return a tuple
            (output_vector, (list of logits), (list of activations))
            or just the output_vector

        use_best: boolean, whether to load the best network (if
            it has been defined) or to use just the current weights

        """
    	# Set iterable variable to input vector
    	x = x_0
    	# Define container for logits and activations
    	z_container = [af.inversegamma(x_0)]
    	x_container = [x_0]
    	# if we want to use the best net out of
    	# the training run, and such net has been defined:
    	if use_best and self.best_net:
    		weights, biases, test_mean_accuracy_DUMMY, epoch_index_DUMMY = self.best_net
    	else:
    		weights, biases = self.weights, self.biases
    	# Loop over all layers and compute the activation
    	# arrays one by one. Note that the first tuple from
    	# this loop computes the activation in layer=1, not layer=0
    	for i,(w,b) in enumerate(zip(weights, biases)):
    		z = af.logit(x,w,b)
    		# when reaching last layer check if you
    		# should use Softmax
    		if self.softmax and i==len(weights)-1:
    			x = af.SoftMax.gamma(z)
    		else:
    			x = self.activation.gamma(z)

    		z_container.append(z)
    		x_container.append(x)

    	# return the output activations
    	if return_layers:
    		return x, z_container, x_container
    	else:
    		return x

    def predict_proba(self, X_0):
        """ Prediction of class probabilities.

        Parameters
        ----------
        X_0 : numpy array, shape = (Nsamples, Nchannels)
                The input samples.

        Returns
        -------
        X_L : numpy array, shape = (Nsamples, Ndim_output)
            The predicted output probabilities.
        """
        X_L = np.zeros((X_0.shape[0], self.output_depth[0]))
        for i, x_ in enumerate(X_0):
            X_L[i] = self.ff(x_, use_best=True)
    	return X_L

    def success_metric(self, y, x_L):
        """
        Given the true (y) and predicted (x_L) output vectors for
        a single sample, return the success metric. Generally, this
        is determined as 1 or 0, but when the MLP is working
        as autoencoder, it is calculated as 1 - rms().
        """
    	if self.autoencoding:
    		sample_success_metric = 1.0 - af.rms(y,x_L)
    	else:
    		sample_success_metric = 1 if y[np.argmax(x_L)]==1 \
            else 0

    	return sample_success_metric

    def sample_metrics(self,y,x_L):
        """
        Given the true (y) and predicted (x_L) output vectors for
        a single sample, return a tuple with the cost and
        success metric
        """
        sample_cost = self.cost_object.J_i(y,x_L)
        sample_success_metric = self.success_metric(y,x_L)
        return sample_cost, sample_success_metric

    def evaluate_test(self):
    	""" Computes perfomance metrics based on test data """
    	# If test data was supplied...
    	if self.test_X is not None and self.test_y is not None:
    		# Initialize aggregators for cost and accuracy
    		sample_cost_agg = 0
    		prediction_success_count = 0
    		# iterate over all samples in the test data set
    		for t in range(len(self.test_X)):
    			# compute the output activation layer (the prediction vector)
    			ypred = self.ff(self.test_X[t])
    			# fetch truth vector
    			ytrue = self.test_y[t]
    			# compute cost and whether the prediction was correct for the
    			# current sample:
    			sample_cost, sample_success_metric = self.sample_metrics(ytrue,ypred)
    			# update aggregators
    			prediction_success_count += sample_success_metric*self.test_norm_coeff[t]
    			sample_cost_agg += sample_cost*self.test_norm_coeff[t]
    		#
    		mean_sample_cost = sample_cost_agg#*1.0/sum(self.test_norm_coeff)
    		mean_accuracy = prediction_success_count#*1.0/sum(self.test_norm_coeff)
    		return mean_sample_cost, mean_accuracy

    	else:
    		print 'No test data supplied'


    def backpropagation(self, x_0, y):
        """ Backpropagation function.

        Parameters
        ----------
        x_0 : numpy array, shape = (Nchannels,)
                The input sample.
        y : numpy array, shape = (Nchannels,)
                The true sample target.

        Returns
        -------
        Tuple that includes:
            z_container: list of logits
            x_container: list of activations
            Jb_grad: list of bias gradients
            Jw_grad: list of weight gradients
            sample_cost: cost for this sample
            sample_success_metric: success metric
        """
        # Feed forward
    	x_L, z_container, x_container = \
            self.ff(x_0, return_layers=True)

    		# Determine final cost, and whether the prediction was successful
    	sample_cost = self.cost_object.J_i(y,x_L)

    	sample_success_metric = self.success_metric(y,x_L)

    	# determine delta_L
    	delta_L = self.cost_object.delta_L(y, x_L)

    	# set iterable to initial value
    	delta_l = delta_L
    	# create container with the output layer
    	delta_l_container = [delta_L]

    	### BACKPROP  ####
    	# iterate backwards to compute each delta_l
    	for u in range(1,self.Nlayers-1):
    		# get the weights from the current layer,
    		# starting from L
    		w = self.weights[-u]
    		# get the activations from the lower layer (current-1)
    		x_prev = x_container[-(u+1)]
    		# compute delta_l for the lower layer
    		delta_l = np.dot(w.T,delta_l)*self.activation.gammaprime(x_prev)
    		#
    		delta_l_container.append(delta_l)
    	#
    	# reverse the delta_l_container, so that
    	# it goes in ascending layer order and matches
    	# the indexing of self.weights
    	delta_l_container = delta_l_container[::-1]

    	# compute the w- and b-gradients
    	Jb_grad = delta_l_container
    	Jw_grad = [delta.reshape(len(delta),1)*x for delta,x \
    			in zip(delta_l_container,x_container[:-1])]

    	return z_container, x_container, Jb_grad, Jw_grad, \
        sample_cost, sample_success_metric

    def fit_single_epoch(self, verbose=False):
        """
        Performs a model fit over one epoch. Returns a tuple
        with cost and accuracy for the training data.
        """
    	# In every epoch we need to randomize the samples
    	# and use them all.
    	# Get a unique random indexing:
    	batch_accuracies = []
    	batch_costs = []
    	random_indices = np.arange(self.Nsamples)
    	np.random.shuffle(random_indices)
        # Iterate over batches
    	for i in range(self.Nbatches):
    		# batches will be defined by contiguous chunks of the randomized index
    		idx_selection = random_indices[i*self.batch_size:(i+1)*self.batch_size]
    		train_X_batch = self.train_X[idx_selection]
    		train_y_batch = self.train_y[idx_selection]
    		train_coeff_batch = self.train_norm_coeff[idx_selection]
    		train_coeff_batch_norm = sum(train_coeff_batch)
    		#
    		Jw_grad_batch_aggregator = [np.zeros(w.shape) for w in self.weights]
    		Jb_grad_batch_aggregator = [np.zeros(b.shape) for b in self.biases]
    		mean_accuracy = 0
    		mean_sample_cost = 0
    		# Iterate over individual samples
    		for k in range(self.batch_size):
    			data_sample = train_X_batch[k]
    			target_sample = train_y_batch[k]
    			coeff_sample = \
                train_coeff_batch[k]/train_coeff_batch_norm
    			z_container, x_container, Jb_grad, Jw_grad,\
                sample_cost, sample_success_metric = \
    					self.backpropagation(data_sample,target_sample)
    			#
    			# update batch aggregator by summing elementwise its array elements with b_brad, which is
    			# of the same shape. Note that we add the sample weight factor here
    			#
    			Jb_grad_batch_aggregator = [ba+b*coeff_sample for ba,b in zip(Jb_grad_batch_aggregator, Jb_grad)]
    			Jw_grad_batch_aggregator = [wa+w*coeff_sample for wa,w in zip(Jw_grad_batch_aggregator, Jw_grad)]
    			mean_sample_cost += sample_cost*coeff_sample
    			mean_accuracy += sample_success_metric*coeff_sample

    		# per batch aggregators
    		batch_accuracies.append(mean_accuracy)
    		batch_costs.append(mean_sample_cost)

    		# at this point, we have gone thru all samples within a batch,
    		# and aggregated the Jw,Jb gradients. We can update coefficients now:
    		if self.tied_weights:
    			# if weights are tied, we will assume this is for autoencoding
    			# and there is only one hidden layer.
    			# The gradients for each weight layer will be different, so we
    			# can just take the average gradient and apply it to both layers,
    			# so that they remain the same
    			w0 = self.weights[0]
    			j0 = Jw_grad_batch_aggregator[0]
    			w1 = self.weights[1]
    			j1 = Jw_grad_batch_aggregator[1].T
    			j_mean = (j0+j1)*1.0/2
    			w0_updated = \
                w0 - self.eta*(j_mean+self.lamb*w0*1.0/self.Nsamples)
    			self.weights = [w0_updated, w0_updated.T]
    		else:
    			self.weights =\
                 [w - self.eta*(dw+self.lamb*w*1.0/self.Nsamples) \
                 for w,dw \
                 in zip(self.weights,Jw_grad_batch_aggregator)]

    		self.biases = [b - self.eta*db for b,db \
            in zip(self.biases,Jb_grad_batch_aggregator)]

    		self.train_cost_container.append(mean_sample_cost)
    		self.train_accuracy_container.append(mean_accuracy)

    	# return epoch-aggregated metrics
    	train_mean_sample_cost = np.mean(batch_costs)
    	train_mean_accuracy = np.mean(batch_accuracies)
    	if verbose:
    		print 'Mean sample cost is %1.3f and the mean accuracy is %1.3f' % (train_mean_sample_cost, train_mean_accuracy)

    	return (train_mean_sample_cost, train_mean_accuracy)


    def fit(self,n_epochs=10):
        """
        Performs a model fit over a number of epochs.

        After each epoch, it also evaluates performance on
        test data, and prints it to screen.

        The training and test performance are saved into lists.
        This function will save the best performing network on the
        test set as self.best_net. After all epochs are completed,
        it is recommended to use the best_net for evaluation, since
        the network might have deviated from its best state. Ideally,
        the last network weights should the best ones, but in practice,
        SGD can move away from minima.
        """
    	for r in range(n_epochs):
            # Fit one epoch, get training cost and accuracy
    		train_mean_sample_cost, train_mean_accuracy =\
             self.fit_single_epoch()
    		if self.test_X is not None:
                # If test data was supplied, evaluate cost
                # and accuracy
    			test_mean_sample_cost, test_mean_accuracy = self.evaluate_test()
    			# determine if this run is the one with best
                # performance. If so, save the network as best_net:
    			if self.test_epoch_accuracies and test_mean_accuracy >= max(self.test_epoch_accuracies):
    				# note that the current epoch has not been appended yet, thus
    				# this length represents the current epoch's index
    				epoch_index = len(self.test_epoch_accuracies)
    				self.best_net = (self.weights, self.biases, test_mean_accuracy, epoch_index)

    			print 'Epoch %s train_accuracy: %1.3f  test_accuracy: %1.3f' % (r,train_mean_accuracy,test_mean_accuracy)
    			self.test_epoch_accuracies.append(test_mean_accuracy)

    		else:
    			print 'Epoch %s train_accuracy: %1.3f' % (r,train_mean_accuracy)
    		#
    		self.train_epoch_accuracies.append(train_mean_accuracy)















    #
