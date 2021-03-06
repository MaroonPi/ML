import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
	"""
	Information on F1 score - https://en.wikipedia.org/wiki/F1_score
	:param real_labels: List[int]
	:param predicted_labels: List[int]
	:return: float
	"""
	assert len(real_labels) == len(predicted_labels)
	npReal = np.array(real_labels)
	npPredicted = np.array(predicted_labels)
	numerator = 2*np.sum(npReal*npPredicted)
	denominator = np.sum(npReal) + np.sum(npPredicted)
	if(denominator==0):
		f1 = 1
	else:
		f1 = numerator/denominator
	#f1 = (2*np.sum(npReal*npPredicted))/(np.sum(npReal)+np.sum(npPredicted))
	return f1


class Distances:
	@staticmethod
	# TODO
	def canberra_distance(point1, point2):
		"""
		:param point1: List[float]
		:param point2: List[float]
		:return: float
		"""
		x = np.array(point1)
		y = np.array(point2)
		numerator = np.absolute(x-y)
		denominator = np.absolute(x)+np.absolute(y)
		div = np.true_divide(numerator, denominator, where=(denominator!=0))
		div = div.astype(float)
		distance = np.sum(div)
		#distance = np.sum(np.absolute(x-y)/(np.absolute(x)+np.absolute(y)))
		return distance

	@staticmethod
	# TODO
	def minkowski_distance(point1, point2):
		"""
		Minkowski distance is the generalized version of Euclidean Distance
		It is also know as L-p norm (where p>=1) that you have studied in class
		For our assignment we need to take p=3
		Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
		:param point1: List[float]
		:param point2: List[float]
		:return: float
		"""
		x = np.array(point1)
		y = np.array(point2)
		distance = (np.sum(np.power(np.absolute(x-y),3)))**(1/3)
		return distance

	@staticmethod
	# TODO
	def euclidean_distance(point1, point2):
		"""
		:param point1: List[float]
		:param point2: List[float]
		:return: float
		"""
		x = np.array(point1)
		y = np.array(point2)
		distance = (np.sum(np.power(x-y,2)))**(1/2)
		return distance

	@staticmethod
	# TODO
	def inner_product_distance(point1, point2):
		"""
		:param point1: List[float]
		:param point2: List[float]
		:return: float
		"""
		x = np.array(point1)
		y = np.array(point2)
		distance = np.dot(x,y)
		return distance

	@staticmethod
	# TODO
	def cosine_similarity_distance(point1, point2):
		"""
		:param point1: List[float]
		:param point2: List[float]
		:return: float
		"""
		x = np.array(point1)
		y = np.array(point2)
		#cosSimilarity = np.dot(x,y)/((np.dot(x,x)**(1/2))*(np.dot(y,y)**(1/2)))
		numerator = np.dot(x,y)
		denominator = (np.dot(x,x)**(1/2))*(np.dot(y,y)**(1/2))
		if(denominator==0):
			distance = 0
		else:
			cosSimilarity = numerator/denominator
			distance = 1 - cosSimilarity
		return distance

	@staticmethod
	# TODO
	def gaussian_kernel_distance(point1, point2):
		"""
		:param point1: List[float]
		:param point2: List[float]
		:return: float
		"""
		x = np.array(point1)
		y = np.array(point2)
		distance = (-1)*np.exp((-1/2)*(np.sum(np.power(x-y,2))))
		return distance

class HyperparameterTuner:
	def __init__(self):
		self.best_k = None
		self.best_distance_function = None
		self.best_scaler = None
		self.best_model = None

	# TODO: find parameters with the best f1 score on validation dataset
	def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
		"""
		In this part, you should try different distance function you implemented in part 1.1, and find the best k.
		Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

		:param distance_funcs: dictionary of distance functions you must use to calculate the distance.
			Make sure you loop over all distance functions for each data point and each k value.
			You can refer to test.py file to see the format in which these functions will be
			passed by the grading script
		:param x_train: List[List[int]] training data set to train your KNN model
		:param y_train: List[int] train labels to train your KNN model
		:param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
			predicted labels and tune k and distance function.
		:param y_val: List[int] validation labels

		Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
		self.best_distance_function and self.best_model respectively.
		NOTE: self.best_scaler will be None

		NOTE: When there is a tie, choose model based on the following priorities:
		Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
		If they have same distance fuction, choose model which has a less k.
		"""
		# You need to assign the final values to these variables
		self.best_k = None
		self.best_distance_function = None
		self.best_model = None
		max_f1 = 0
		function_order = ['canberra','minkowski','euclidean','gaussian','inner_prod','cosine_dist']
		for function in distance_funcs:
			for k in range(1,30,2):
				model = KNN(k,distance_funcs[function])
				model.train(x_train,y_train)
				predicted_values = model.predict(x_val)
				f1 = f1_score(y_val,predicted_values)
				if(self.best_model==None):
					self.best_model = model
					self.best_distance_function = function
					self.best_k = k
					max_f1 = f1
				else:
					#Comparing f1 scores
					if(f1>max_f1):
						max_f1 = f1
						self.best_model = model
						self.best_distance_function = function
						self.best_k = k
					elif(f1==max_f1):
						#Comparing distance functions
						if(function_order.index(self.best_distance_function)>function_order.index(function)):
							self.best_model = model
							self.best_distance_function = function
							self.best_k = k
						elif(function_order.index(self.best_distance_function)==function_order.index(function)):
							#Comparing k values
							if(k<self.best_k):
								self.best_model = model
								self.best_distance_function = function
								self.best_k = k

	# TODO: find parameters with the best f1 score on validation dataset, with normalized data
	def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
		"""
		This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
		tune k and disrance function, you need to create the normalized data using these two scalers to transform your
		data, both training and validation. Again, we will use f1-score to compare different models.
		Here we have 3 hyperparameters i.e. k, distance_function and scaler.

		:param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
			loop over all distance function for each data point and each k value.
			You can refer to test.py file to see the format in which these functions will be
			passed by the grading script
		:param scaling_classes: dictionary of scalers you will use to normalized your data.
		Refer to test.py file to check the format.
		:param x_train: List[List[int]] training data set to train your KNN model
		:param y_train: List[int] train labels to train your KNN model
		:param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
			labels and tune your k, distance function and scaler.
		:param y_val: List[int] validation labels

		Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
		self.best_distance_function, self.best_scaler and self.best_model respectively

		NOTE: When there is a tie, choose model based on the following priorities:
		For normalization, [min_max_scale > normalize];
		Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
		If they have same distance function, choose model which has a less k.
		"""

		# You need to assign the final values to these variables
		self.best_k = None
		self.best_distance_function = None
		self.best_scaler = None
		self.best_model = None
		max_f1 = 0
		function_order = ['canberra','minkowski','euclidean','gaussian','inner_prod','cosine_dist']
		scaler_order = ['min_max_scale','normalize']
		for scaler in scaling_classes:
			for function in distance_funcs:
				for k in range(1,30,2):
					#Scale the data with the scaler
					currentScaler = scaling_classes[scaler]()
					scaled_x_train = currentScaler(x_train)
					scaled_x_val = currentScaler(x_val)
					model = KNN(k,distance_funcs[function])
					model.train(scaled_x_train,y_train)
					predicted_values = model.predict(scaled_x_val)
					f1 = f1_score(y_val,predicted_values)
					if(self.best_model==None):
						self.best_model = model
						self.best_distance_function = function
						self.best_k = k
						self.best_scaler = scaler
						max_f1 = f1
					else:
						#Comparing f1 scores
						if(f1>max_f1):
							max_f1 = f1
							self.best_model = model
							self.best_distance_function = function
							self.best_k = k
							self.best_scaler = scaler
						elif(f1==max_f1):
							#Comparing scalers
							if(scaler_order.index(self.best_scaler)>scaler_order.index(scaler)):
								self.best_model = model
								self.best_distance_function = function
								self.best_k = k
								self.best_scaler = scaler
							elif(scaler_order.index(self.best_scaler)==scaler_order.index(scaler)):
								#Comparing distance functions
								if(function_order.index(self.best_distance_function)>function_order.index(function)):
									self.best_model = model
									self.best_distance_function = function
									self.best_k = k
									self.best_scaler = scaler
								elif(function_order.index(self.best_distance_function)==function_order.index(function)):
									#Comparing k values
									if(k<=self.best_k):
										self.best_model = model
										self.best_distance_function = function
										self.best_k = k
										self.best_scaler = scaler


class NormalizationScaler:
	def __init__(self):
		pass

	# TODO: normalize data
	def __call__(self, features):
		"""
		Normalize features for every sample

		Example
		features = [[3, 4], [1, -1], [0, 0]]
		return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

		:param features: List[List[float]]
		:return: List[List[float]]
		"""
		features = np.array(features,dtype=float)
		for row in range(features.shape[0]):
		    if(np.any(features[row,:])):
		        numerator = features[row,:]
		        denominator = np.dot(features[row,:],features[row,:])**(1/2)
		        features[row,:] = numerator/denominator
		    else:
		        features[row,:] = np.zeros_like(features[row,:])
		normalized = features.tolist()
		return normalized


class MinMaxScaler:
	"""
	Please follow this link to know more about min max scaling
	https://en.wikipedia.org/wiki/Feature_scaling
	You should keep some states inside the object.
	You can assume that the parameter of the first __call__
	will be the training set.

	Hints:
		1. Use a variable to check for first __call__ and only compute
			and store min/max in that case.

	Note:
		1. You may assume the parameters are valid when __call__
			is being called the first time (you can find min and max).

	Example:
		train_features = [[0, 10], [2, 0]]
		test_features = [[20, 1]]

		scaler1 = MinMaxScale()
		train_features_scaled = scaler1(train_features)
		# train_features_scaled should be equal to [[0, 1], [1, 0]]

		test_features_scaled = scaler1(test_features)
		# test_features_scaled should be equal to [[10, 0.1]]

		new_scaler = MinMaxScale() # creating a new scaler
		_ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
		test_features_scaled = new_scaler(test_features)
		# now test_features_scaled should be [[20, 1]]

	"""

	def __init__(self):
		self.firstRun = True
		self.maxValues = []
		self.minValues = []

	def __call__(self, features):
		"""
		normalize the feature vector for each sample . For example,
		if the input features = [[2, -1], [-1, 5], [0, 0]],
		the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

		:param features: List[List[float]]
		:return: List[List[float]]
		"""
		features = np.array(features,dtype=float)
		if(self.firstRun):
			self.firstRun = False
			for column in range(features.shape[1]):
				min = np.amin(features[:,column])
				self.minValues.append(min)
				max = np.amax(features[:,column])
				self.maxValues.append(max)
				if((max-min)==0):
					features[:,column] = features[:,column]-min
				else:
					features[:,column] = (features[:,column]-min)/(max-min)
		else:
			for column in range(features.shape[1]):
				if((self.maxValues[column]-self.minValues[column])==0):
					features[:,column] = features[:,column]-self.minValues[column]
				else:
					features[:,column] = (features[:,column]-self.minValues[column])/(self.maxValues[column]-self.minValues[column])
		normalized = features.tolist()
		return normalized
