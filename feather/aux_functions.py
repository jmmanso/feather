import numpy as np


def logit(x_prev, w, b):
    """ Logit value of neuron.

    Parameters
    ----------
    w: w[k,j] array of weights, relating neuron
      k in layer l with neuron j in layer l-1

    b: b[k] bias of neuron k in layer l

    x_prev: activation array from layer l-1
    """
    return np.dot(w, x_prev) + b

def gamma(z):
	return 1.0/(1.0+np.exp(-z))

def gammaprime(z):
    # derivative of gamma
	expz = np.exp(-z)
	return expz/(1.0+expz)**2

def inversegamma(x):
	return -np.log((x-1.0)/x)

def rms(vec1,vec2):
	return np.sqrt(sum((vec1-vec2)**2)/len(vec1))




class L_cost:
    """
    Cost function per sample
    """
    def __init__(self, cost_type, last_activation_type):
    	self.cost_type = cost_type
    	self.last_activation_type = last_activation_type

    	if cost_type == 'Xentropy':
    		self.cost_class = Xentropy
    	elif cost_type == 'Gaussian':
    		self.cost_class = Gaussian
    	else:
    		print 'error: not valid cost type'
    	#
    	if last_activation_type == 'softmax':
    		self.last_activation_class = SoftMax

    	elif last_activation_type == 'tanh':
    		self.last_activation_class = Tanh

    	elif last_activation_type == 'sigmoid':
    		self.last_activation_class = Sigmoid

    	elif last_activation_type == 'relu':
    		self.last_activation_class = ReLu

    	else:
    		print 'error: not valid activation type'

    #
    def delta_L(self, ytrue_vector, ypred_vector):

    	return self.cost_class.dJdxL(ytrue_vector, ypred_vector)*\
    	self.last_activation_class.gammaprime(ypred_vector)

    def J_i(self,ytrue_vector, ypred_vector):
    	return self.cost_class.J_i(ytrue_vector, ypred_vector)



class Xentropy:

	@staticmethod
	def J_i(ytrue_vector, ypred_vector):

		return sum(-ytrue_vector*np.log10(ypred_vector) -\
		 (1.0-ytrue_vector)*np.log10(1.0-ypred_vector))

	@staticmethod
	def dJdxL(ytrue_vector, ypred_vector):
		return (ytrue_vector - 1.0)/(ypred_vector - 1.0) - ytrue_vector/ypred_vector


class Gaussian:

	@staticmethod
	def J_i(ytrue_vector, ypred_vector):

		return 0.5*sum((ytrue_vector-ypred_vector)**2)

	@staticmethod
	def dJdxL(ytrue_vector, ypred_vector):
		return ypred_vector - ytrue_vector



class Activation:

	def __init__(self,activation_type):

		self.activation_type = activation_type

		if activation_type == 'softmax':
			self.activation_class = SoftMax

		elif activation_type == 'tanh':
			self.activation_class = Tanh

		elif activation_type == 'sigmoid':
			self.activation_class = Sigmoid

		elif activation_type == 'relu':
			self.activation_class = ReLu

		else:
			print 'error: not valid activation type'

	def gamma(self, zvec):
		return self.activation_class.gamma(zvec)

	def gammaprime(self, xvec):
		return self.activation_class.gammaprime(xvec)


class SoftMax:

	@staticmethod
	def gamma(z_vec):
		""" Takes a vector of logits and returns
		an activation vector of the same length """
		norm = sum(np.exp(-z_vec))
		return np.exp(-z_vec)*1.0/norm

	@staticmethod
	def gammaprime(x_vec):
		return x_vec*(x_vec - 1.0)


class Sigmoid:

	@staticmethod
	def gamma(z_vec):
		return 1.0/(1.0+np.exp(-z_vec))

	@staticmethod
	def gammaprime(x_vec):
		return x_vec*(1.0 - x_vec)

class Tanh:

	@staticmethod
	def gamma(z_vec):
		return 2.0/(1.0+np.exp(-2*z_vec)) - 1.0

	@staticmethod
	def gammaprime(x_vec):
		return 1.0 - x_vec**2



class ReLu:

	@staticmethod
	def gamma(z_vec):
		k = np.greater_equal(z_vec,z_vec*0).astype('int')
		p = np.logical_not(k)*(0.01)
		return (k+p)*z_vec


	@staticmethod
	def gammaprime(x_vec):
		k = np.greater_equal(x_vec,x_vec*0).astype('int')
		p = np.logical_not(k)*(-0.01)
		return (k+p)











#
