from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        self.alpha = None
        self.beta = None
        self.seqProb = None
        self.posProb = None
        self.Psi = None

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for t in range(L):
            for state in self.state_dict:
                state_index = self.state_dict[state]
                if(t==0):
                    obs_zero_index = self.obs_dict[Osequence[0]]
                    alpha[state_index,0] = self.pi[state_index] * self.B[state_index,obs_zero_index]
                else:
                    obs_index = self.obs_dict[Osequence[t]]
                    summationTerm = np.sum(alpha[:,t-1]*self.A[:,state_index])
                    alpha[state_index,t] = self.B[state_index,obs_index] * summationTerm
        self.alpha = alpha
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for t in range(L-1,-1,-1):
            for state in self.state_dict:
                state_index = self.state_dict[state]
                if(t==L-1):
                    beta[state_index,t] = 1
                else:
                    obs_index = self.obs_dict[Osequence[t+1]]
                    beta[state_index,t] = np.sum(self.A[state_index,:]*self.B[:,obs_index]*beta[:,t+1])
        self.beta = beta
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        L = len(Osequence)
        prob = np.sum(self.alpha[:,L-1])
        self.seqProb = prob
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        prob = (self.alpha*self.beta)/self.seqProb
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        ###################################################
        for t in range(L-1):
            obs_index = self.obs_dict[Osequence[t+1]]
            for s in range(S):
                for prev_S in range(S):
                    prob[s,prev_S,t] = self.alpha[s,t]*self.A[s,prev_S]*self.B[prev_S,obs_index]*self.beta[prev_S,t+1]/self.seqProb
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        sDelta = np.zeros([S,L])
        lDelta = np.zeros([S,L],dtype=int)
        for state in self.state_dict:
            state_index = self.state_dict[state]
            obs_zero_index = self.obs_dict[Osequence[0]]
            sDelta[state_index,0] = self.pi[state_index] * self.B[state_index,obs_zero_index]
        for t in range(1,L):
            for state in self.state_dict:
                state_index = self.state_dict[state]
                obs_index = self.obs_dict[Osequence[t]]
                maxTerm = self.A[0,state_index]*sDelta[0,t-1]
                for column in range(1,S):
                    if(maxTerm<(self.A[column,state_index]*sDelta[column,t-1])):
                        maxTerm = self.A[column,state_index]*sDelta[column,t-1]
                sDelta[state_index,t] = self.B[state_index,obs_index] * maxTerm
                lDelta[state_index,t] = np.argmax((self.A[:,state_index]*sDelta[:,t-1]))
                """
                maxTerm = np.amax(self.A[:,state_index]*sDelta[:,t-1])
                print(maxTerm)
                sDelta[state_index,t] = self.B[state_index,t] * maxTerm
                lDelta[state_index,t] = np.argmax((self.A[:,state_index]*sDelta[:,t-1]))
                """
        invertedStateDict = dict([[v,k] for k,v in self.state_dict.items()])
        reqIndex = np.argmax(sDelta[:,L-1],axis=0)
        path.insert(0,invertedStateDict[reqIndex])
        for t in range(L-1,0,-1):
            reqIndex = lDelta[reqIndex,t]
            path.insert(0,invertedStateDict[reqIndex])
        return path
