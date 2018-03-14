"""
Authors: Bibek Paudel and Thilo Haas

Theano implementation of following matrix factorization methods:
 - Logistic Matrix Factorization
 - TC-MF
 - TC-MF1
 - S-MF

For more details, please consult this paper:
[1] Paudel, Bibek, Thilo Haas, and Abraham Bernstein. "Fewer Flops at the Top: Accuracy, Diversity, and Regularization in Two-Class Collaborative Filtering." Proceedings of the Eleventh ACM Conference on Recommender Systems. ACM, 2017.

Original Logistic Matrix Factorization Python Code available at: https://github.com/MrChrisJohnson/logistic-mf/blob/master/logistic_mf.py and adjusted to fit
"""
import numpy as np
from . import base
import scipy

import theano.tensor as T
import theano
from adagrad import adagrad_update

class LogisticMF():

    def __init__(self, n_components=1, a=0.6, gamma=1.0,
                 iterations=30):
        self.n_components = n_components
        self.iterations = iterations
        self.reg = a
        self.gamma = gamma
        self.counts = None
        self.estimate_matrix = None
        self.recommendation_matrix = None

    def get_model_data(self):
        return {
            'user_vectors': self.user_vectors,
            'item_vectors': self.item_vectors,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
        }

    def set_model_data(self, data, trainset, testset):
        self.user_vectors = data['user_vectors']
        self.item_vectors = data['item_vectors']
        self.user_biases = data['user_biases']
        self.item_biases = data['item_biases']
        self.trainset = np.array(trainset)

        self.counts = base.convert_ratings_to_array(trainset)
        self.num_users = self.counts.shape[0]
        self.num_items = self.counts.shape[1]

    def fit(self, trainset, y=None):

        self.trainset = np.array(trainset)
        X = base.convert_ratings_to_array(trainset)
        self.num_users = X.shape[0]
        self.num_items = X.shape[1]
        self.counts = X
        self.estimate_matrix = None
        self.recommendation_matrix = None

        self.ones = np.ones((self.num_users, self.num_items))
        user_vectors = theano.shared(np.random.normal(size=(self.num_users, self.n_components)))
        item_vectors = theano.shared(np.random.normal(size=(self.n_components, self.num_items)))
        user_biases = theano.shared(np.random.normal(size=(self.num_users)).reshape(self.num_users, 1), broadcastable=(False, True))
        item_biases = theano.shared(np.random.normal(size=(self.num_items)).reshape(1, self.num_items), broadcastable=(True, False))

        R = T.matrix("R")
        error = T.sum(R * (user_vectors.dot(item_vectors) + user_biases + item_biases) -
                      (1 + R) * T.log(1 + T.exp(user_vectors.dot(item_vectors) + user_biases + item_biases)))
        regularization = (self.reg / 2.0) * (T.sum(T.sqr(user_vectors)) + T.sum(T.sqr(item_vectors)))
        cost = error - regularization

        train_users = theano.function(inputs=[R],
                                  outputs=cost,
                                  updates=adagrad_update(cost, [user_vectors, user_biases], self.gamma))
        train_items = theano.function(inputs=[R],
                                  outputs=cost,
                                  updates=adagrad_update(cost, [item_vectors, item_biases], self.gamma))
        for iter in xrange(self.iterations):
            c = train_users(self.counts)
            c = train_items(self.counts)

        self.user_vectors = user_vectors.get_value()
        self.item_vectors = item_vectors.get_value().T
        self.user_biases = user_biases.get_value().reshape((self.num_users, 1))
        self.item_biases = item_biases.get_value().reshape((self.num_items, 1))

        return self

    def get_estimate_matrix(self):

        if self.estimate_matrix is None:
            A = np.dot(self.user_vectors, self.item_vectors.T)
            A += self.user_biases
            A += self.item_biases.T
            A = scipy.special.expit(A)

            self.estimate_matrix = np.array(A)
        return self.estimate_matrix

    def get_recommendation_matrix(self):
        if self.recommendation_matrix is None:
            self.recommendation_matrix = self.get_estimate_matrix()
            self.recommendation_matrix[self.counts != 0.0] = 0.0

        return self.recommendation_matrix

    def get_trainset(self):
        return self.trainset

    def predict(self, user, item):
        prediction = self.get_estimate_matrix()

        return prediction[user][item]

class TCMF1(LogisticMF):

    def fit(self, trainset, y=None):
        self.trainset = np.array(trainset)
        X = base.convert_ratings_to_array(trainset)
        self.num_users = X.shape[0]
        self.num_items = X.shape[1]
        self.counts = X
        self.estimate_matrix = None
        self.recommendation_matrix = None

        posCounts = np.copy(self.counts)
        posCounts[self.counts < 0.0] = 0

        self.ones = np.ones((self.num_users, self.num_items))
        user_vectors = theano.shared(np.random.normal(size=(self.num_users, self.n_components)))
        item_vectors = theano.shared(np.random.normal(size=(self.n_components, self.num_items)))
        user_biases = theano.shared(np.random.normal(size=(self.num_users)).reshape(self.num_users, 1), broadcastable=(False, True))
        item_biases = theano.shared(np.random.normal(size=(self.num_items)).reshape(1, self.num_items), broadcastable=(True, False))

        R = T.matrix("R")
        error = T.sum(R * (user_vectors.dot(item_vectors) + user_biases + item_biases) -
                      T.log(1 + T.exp(user_vectors.dot(item_vectors) + user_biases + item_biases)))
        regularization = (self.reg / 2.0) * (T.sum(T.sqr(user_vectors)) + T.sum(T.sqr(item_vectors)))
        cost = error - regularization

        train_users = theano.function(inputs=[R],
                                  outputs=cost,
                                  updates=adagrad_update(cost, [user_vectors, user_biases], self.gamma))
        train_items = theano.function(inputs=[R],
                                  outputs=cost,
                                  updates=adagrad_update(cost, [item_vectors, item_biases], self.gamma))
        for iter in xrange(self.iterations):
            c = train_users(posCounts)
            c = train_items(posCounts)

        self.user_vectors = user_vectors.get_value()
        self.item_vectors = item_vectors.get_value().T
        self.user_biases = user_biases.get_value().reshape((self.num_users, 1))
        self.item_biases = item_biases.get_value().reshape((self.num_items, 1))

        return self

class TCMF(LogisticMF):
    def fit(self, trainset, y=None):
        self.trainset = np.array(trainset)
        X = base.convert_ratings_to_array(trainset)
        self.num_users = X.shape[0]
        self.num_items = X.shape[1]
        self.counts = X
        self.estimate_matrix = None
        self.recommendation_matrix = None

        posCounts = np.copy(self.counts)
        posCounts[self.counts < 0.0] = 0.0
        nonzeroCounts = np.copy(self.counts)
        nonzeroCounts[self.counts != 0.0] = 1.0

        self.ones = np.ones((self.num_users, self.num_items))
        user_vectors = theano.shared(np.random.normal(size=(self.num_users, self.n_components)))
        item_vectors = theano.shared(np.random.normal(size=(self.n_components, self.num_items)))
        user_biases = theano.shared(np.random.normal(size=(self.num_users)).reshape(self.num_users, 1), broadcastable=(False, True))
        item_biases = theano.shared(np.random.normal(size=(self.num_items)).reshape(1, self.num_items), broadcastable=(True, False))

        RP = T.matrix("RP") #posCounts
        RNZ = T.matrix("RNZ") #nonzeroCounts
        error = T.sum(RP * (user_vectors.dot(item_vectors) + user_biases + item_biases) -
                      RNZ * T.log(1 + T.exp(user_vectors.dot(item_vectors) + user_biases + item_biases)))
        regularization = (self.reg / 2.0) * (T.sum(T.sqr(user_vectors)) + T.sum(T.sqr(item_vectors)))
        cost = error - regularization

        train_users = theano.function(inputs=[RP, RNZ],
                                  outputs=cost,
                                  updates=adagrad_update(cost, [user_vectors, user_biases], self.gamma))
        train_items = theano.function(inputs=[RP, RNZ],
                                  outputs=cost,
                                  updates=adagrad_update(cost, [item_vectors, item_biases], self.gamma))
        for iter in xrange(self.iterations):
            c = train_users(posCounts, nonzeroCounts)
            c = train_items(posCounts, nonzeroCounts)

        self.user_vectors = user_vectors.get_value()
        self.item_vectors = item_vectors.get_value().T
        self.user_biases = user_biases.get_value().reshape((self.num_users, 1))
        self.item_biases = item_biases.get_value().reshape((self.num_items, 1))

        return self

class SMF():

    def __init__(self, n_components=1, reg_l2=0.001, reg_l1=0.01, pos_gamma=1.0, neg_gamma=1.0, iterations=30):
        self.n_components = n_components
        self.iterations = iterations
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.pos_gamma = pos_gamma
        self.neg_gamma = neg_gamma
        self.pos_recommendation_matrix = None
        self.neg_recommendation_matrix = None

    def get_model_data(self):
        return {
            'user_vectors': self.user_vectors,
            'pos_item_vectors': self.pos_item_vectors,
            'neg_item_vectors': self.neg_item_vectors,
            'user_biases': self.user_biases,
            'pos_item_biases': self.pos_item_biases,
            'neg_item_biases': self.neg_item_biases
        }

    def set_model_data(self, data, trainset, testset):
        self.user_vectors = data['user_vectors']
        self.pos_item_vectors = data['pos_item_vectors']
        self.neg_item_vectors = data['neg_item_vectors']
        self.user_biases = data['user_biases']
        self.pos_item_biases = data['pos_item_biases']
        self.neg_item_biases = data['neg_item_biases']

        self.counts = base.convert_ratings_to_array(trainset)
        self.num_users = self.counts.shape[0]
        self.num_items = self.counts.shape[1]

    def fit(self, trainset, y=None):
        self.trainset = np.array(trainset)
        X = base.convert_ratings_to_array(trainset)
        self.num_users = X.shape[0]
        self.num_items = X.shape[1]
        self.counts = X
        self.pos_recommendation_matrix = None
        self.neg_recommendation_matrix = None
        self.ones = np.ones((self.num_users, self.num_items))

        posCounts = np.copy(self.counts)
        posCounts[self.counts < 0.0] = 0
        negCounts = np.copy(self.counts)
        negCounts[self.counts > 0.0] = 0
        negCounts[negCounts < 0.0] = 1.0

        user_vectors = theano.shared(np.random.normal(size=(self.num_users, self.n_components)))
        pos_item_vectors = theano.shared(np.random.normal(size=(self.n_components, self.num_items)))
        neg_item_vectors = theano.shared(np.random.normal(size=(self.n_components, self.num_items)))
        user_biases = theano.shared(np.random.normal(size=(self.num_users)).reshape(self.num_users, 1),
                                    broadcastable=(False, True))
        pos_item_biases = theano.shared(np.random.normal(size=(self.num_items)).reshape(1, self.num_items),
                                    broadcastable=(True, False))
        neg_item_biases = theano.shared(np.random.normal(size=(self.num_items)).reshape(1, self.num_items),
                                        broadcastable=(True, False))
        RP = T.matrix("RP")  # posCounts
        RN = T.matrix("RN")  # negCounts
        error = T.sum(RP * (user_vectors.dot(pos_item_vectors) + user_biases + pos_item_biases) +
                      RN * (user_vectors.dot(neg_item_vectors) + user_biases + neg_item_biases) -
                      T.log(1 + T.exp(user_vectors.dot(pos_item_vectors) + user_biases + pos_item_biases) +
                            T.exp(user_vectors.dot(neg_item_vectors) + user_biases + neg_item_biases)))
        regularization_l2 = (self.reg_l2 / 2.0) * (T.sum(T.sqr(user_vectors)) +
                                             T.sum(T.sqr(pos_item_vectors)) +
                                             T.sum(T.sqr(neg_item_vectors)))
        regularization_l1 = self.reg_l1 * abs(T.sum(user_vectors) + \
                            T.sum(pos_item_vectors) + \
                            T.sum(neg_item_vectors))

        cost = error - regularization_l2 - regularization_l1

        train_users = theano.function(inputs=[RP, RN],
                                      outputs=cost,
                                      updates=adagrad_update(cost, [user_vectors, user_biases], self.pos_gamma))
        train_pos_items = theano.function(inputs=[RP, RN],
                                      outputs=cost,
                                      updates=adagrad_update(cost,
                                                             [pos_item_vectors, pos_item_biases],
                                                             self.pos_gamma))
        train_neg_items = theano.function(inputs=[RP, RN],
                                          outputs=cost,
                                          updates=adagrad_update(cost,
                                                                 [neg_item_vectors, neg_item_biases],
                                                                 self.neg_gamma))
        for iter in xrange(self.iterations):
            c = train_users(posCounts, negCounts)
            c = train_pos_items(posCounts, negCounts)
            c = train_neg_items(posCounts, negCounts)

        self.user_vectors = user_vectors.get_value()
        self.pos_item_vectors = pos_item_vectors.get_value().T
        self.neg_item_vectors = neg_item_vectors.get_value().T
        self.user_biases = user_biases.get_value().reshape((self.num_users, 1))
        self.pos_item_biases = pos_item_biases.get_value().reshape((self.num_items, 1))
        self.neg_item_biases = neg_item_biases.get_value().reshape((self.num_items, 1))

        return self

    def get_trainset(self):
        return self.trainset

    def get_pos_exp(self):
        return np.exp(np.dot(self.user_vectors, self.pos_item_vectors.T) + self.user_biases + self.pos_item_biases.T)

    def get_neg_exp(self):
        return np.exp(np.dot(self.user_vectors, self.neg_item_vectors.T) + self.user_biases + self.neg_item_biases.T)

    def get_recommendation_matrix(self):
        if self.pos_recommendation_matrix is None:
            np.seterr(over='warn', under='warn')
            pos_exp = self.get_pos_exp()
            neg_exp = self.get_neg_exp()
            self.pos_recommendation_matrix = pos_exp / (self.ones + pos_exp + neg_exp)
            self.neg_recommendation_matrix = neg_exp / (self.ones + pos_exp + neg_exp)
        return self.pos_recommendation_matrix, self.neg_recommendation_matrix

    def predict(self, user, item):
        pos_recommendation_matrix, neg_recommendation_matrix = self.get_recommendation_matrix()

        pos = pos_recommendation_matrix[user][item]
        neg = neg_recommendation_matrix[user][item]
        unknown = 1.0 - pos - neg

        if unknown > pos and unknown > neg:
            return 0.0

        return pos

        #if pos > neg and pos > unknown:
        #    return 1.0
        #elif neg > pos and neg > unknown:
        #    return -1.0
        #return 0.0

    def apply_l1_penalty(self, w_vector, q):
        return w_vector

    def l1_regularizer(self, w):
        l1grad = np.sign(w)
        #l1grad[l1grad == 0.0] = 1.0

        return self.reg_l1 * l1grad

    def l1_projection(self, w):
        """Project to L1-ball, as described by Duchi et al. [ICML '08]."""
        z = 1.0 / (self.reg * self.reg)
        if np.linalg.norm(w, 1) <= z:
            return w

        mu = -np.sort(-w)
        cs = np.cumsum(mu)
        rho = -1
        for j in range(len(w)):
            if mu[j] - (1.0 / (j + 1)) * (cs[j] - z) > 0:
                rho = j
        theta = (1.0 / (rho + 1)) * (cs[rho] - z)

        return np.sign(w) * np.fmax(w - theta, 0) - w
        #return -(np.sign(w) * np.fmax(w - theta, 0) - w)

    def l1penalty(w, q_data_ptr, x_ind_ptr, xnnz, u):
        """Apply the L1 penalty to each updated feature
        This implements the truncated gradient approach by
        [Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009].
        w WeightVector
        double q_data_ptr
        int x_ind_ptr
        int xnnz
        double u
        """
        z = 0.0
        j = 0
        idx = 0
        wscale = 1.0
        w_data_ptr = w.w_data_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            z = w_data_ptr[idx]
            if wscale * w_data_ptr[idx] > 0.0:
                w_data_ptr[idx] = max(
                    0.0, w_data_ptr[idx] - ((u + q_data_ptr[idx]) / wscale))

            elif wscale * w_data_ptr[idx] < 0.0:
                w_data_ptr[idx] = min(
                    0.0, w_data_ptr[idx] + ((u - q_data_ptr[idx]) / wscale))

            q_data_ptr[idx] += wscale * (w_data_ptr[idx] - z)
