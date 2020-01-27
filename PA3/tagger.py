import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	###################################################
	state_dict = {}
	S = len(tags)
	pi = np.zeros([S])
	A = np.zeros([S,S])
	stateIndex = 0
	for tag in tags:
		state_dict[tag] = stateIndex
		stateIndex += 1
	initStateCountDict = {}
	for sentence in train_data:
		if(sentence.tags[0] in initStateCountDict):
			initStateCountDict[sentence.tags[0]] += 1
		else:
			initStateCountDict[sentence.tags[0]] = 1
	for tag in initStateCountDict:
		state_index = state_dict[tag]
		pi[state_index] = initStateCountDict[tag]/len(train_data)
	transitionCountDict = {}
	for state in state_dict:
		transitionCountDict[state] = {}
		for nextState in state_dict:
			transitionCountDict[state][nextState] = 0
	for sentence in train_data:
		for i in range(len(sentence.tags)-1):
			prevTag = sentence.tags[i]
			nextTag = sentence.tags[i+1]
			transitionCountDict[prevTag][nextTag] += 1
	for prevState in transitionCountDict:
		prevStateIndex = state_dict[prevState]
		totalPrevCount = sum(transitionCountDict[prevState].values())
		for nextState in transitionCountDict[prevState]:
			nextStateIndex = state_dict[nextState]
			A[prevStateIndex][nextStateIndex] = transitionCountDict[prevState][nextState]/totalPrevCount
	obs_dict = {}
	obsIndex = 0
	for sentence in train_data:
		for word in sentence.words:
			if(word not in obs_dict):
				obs_dict[word] = obsIndex
				obsIndex += 1
	num_obs = len(obs_dict)
	B = np.zeros([S,num_obs])
	obsCountDict = {}
	for state in state_dict:
		obsCountDict[state] = {}
		for obs in obs_dict:
			obsCountDict[state][obs] = 0
	tagCountDict = {}
	for state in state_dict:
		tagCountDict[state] = 0
	for sentence in train_data:
		for i in range(len(sentence.words)):
			tag = sentence.tags[i]
			word = sentence.words[i]
			obsCountDict[tag][word] += 1
			tagCountDict[tag] += 1
	for state in obsCountDict:
		state_index = state_dict[state]
		for obs in obsCountDict[state]:
			obs_index = obs_dict[obs]
			B[state_index][obs_index] = obsCountDict[state][obs]/tagCountDict[state]
	model = HMM(pi,A,B,obs_dict,state_dict)
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	###################################################
	S = model.B.shape[0]
	newObsIndex = len(model.obs_dict)
	extraProbArray = np.full((S,1),0.000001)
	for sentence in test_data:
		for word in sentence.words:
			if(word not in model.obs_dict):
				model.obs_dict[word] = newObsIndex
				model.B = np.concatenate((model.B,extraProbArray),axis=1)
				newObsIndex += 1
	for sentence in test_data:
		tagged_sentence = model.viterbi(sentence.words)
		tagging.append(tagged_sentence)
	return tagging
