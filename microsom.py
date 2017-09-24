from collections import OrderedDict
import theano
import theano.tensor as T
import theano.gradient as G
import numpy as np
from lasagne.updates import sgd
from enum import Enum


class HistoryLevel(Enum):
    NONE = 0
    EPOCHS = 1
    STEPS = 2


class BasicSOM(object):
    def __init__(self, weights, neurons_topology,
                 learning_rate=0.1, learning_rate_decay=0.985,
                 collaboration_sigma=1.0, collaboration_sigma_decay=0.95, verbosity=2):

        self._verbosity = verbosity
        self._history = []
        self.neurons_number = weights.shape[0]
        self.W_shar_mat = theano.shared(weights)
        self.D_shar_mat = theano.shared(neurons_topology)

        self.collaboration_sigma = theano.shared(collaboration_sigma)
        self.collaboration_sigma_decay = collaboration_sigma_decay

        self.x_row = T.vector("exemplar")
        self.x_mat = T.matrix("batch")

        self.learning_rate = theano.shared(learning_rate)
        self.learning_rate_decay = learning_rate_decay

        self.distance_from_y_row = ((T.sub(self.W_shar_mat, self.x_row)**2).sum(axis=1))
        self.closest_neuron_idx = T.argmin(self.distance_from_y_row)
        self.distances_from_closest_neuron = self.D_shar_mat[self.closest_neuron_idx]
        self.affinities_to_closest_neuron = T.exp(-self.distances_from_closest_neuron/(self.collaboration_sigma)**2)

        self.smoothed_distances_from_closest_neuron = T.mul(self.distance_from_y_row, G.disconnected_grad(self.affinities_to_closest_neuron))
        self.cost_scal = self.smoothed_distances_from_closest_neuron.sum()

        self.updates = sgd(self.cost_scal, [self.W_shar_mat], learning_rate=self.learning_rate)
        self.update_neurons = theano.function([self.x_row], self.cost_scal, updates=self.updates)

    def post_epoch(self, epoch):
        self.collaboration_sigma.set_value(self.collaboration_sigma.eval()*self.collaboration_sigma_decay)
        self.learning_rate.set_value(self.learning_rate.eval()*self.learning_rate_decay)

    def fit(self, X, number_of_epochs, with_history=HistoryLevel.NONE):
        if with_history != HistoryLevel.NONE:
            self._history = []
        order = list(range(len(X)))
        for epoch in range(number_of_epochs):
            print("SOM: Starting epoch {}".format(epoch+1))
            accumulated_cost = 0
            np.random.shuffle(order)
            for i in order:
                if with_history == HistoryLevel.STEPS:
                    self._history.append({
                        "W" : self.W_shar_mat.eval(),
                        "x" : X[i]
                    })
                accumulated_cost += self.update_neurons(X[i])
            self.post_epoch(epoch)
            if with_history == HistoryLevel.EPOCHS:
                self._history.append({
                    "W" : self.W_shar_mat.eval()
                })
            print("SOM: Ending epoch {} with mean cost={}, lr={}, c_sigm={}".format(
                epoch, accumulated_cost/len(X), self.learning_rate.eval(), self.collaboration_sigma))

    def get_neuron_weights(self):
        return self.W_shar_mat.eval()

    def get_x_history(self):
        if self._history and len(self._history) and "x" in self._history[0]:
            return [H["x"] for H in self._history]
        return []

    def get_w_history(self):
        if self._history and len(self._history):
            return [H["W"] for H in self._history]
        return []


class WinnerRelaxingSOM(BasicSOM):
    def __init__(self, weights, neurons_topology, relaxing_factor=-0.5, **kwargs):
        super(WinnerRelaxingSOM, self).__init__(weights, neurons_topology, **kwargs)
        self.wr_relaxing_factor = relaxing_factor
        self.wr_relaxing_member = (
            self.smoothed_distances_from_closest_neuron.sum()
            - self.smoothed_distances_from_closest_neuron[self.closest_neuron_idx]
        )
        self.cost_scal += self.wr_relaxing_factor*self.learning_rate*T.mul(self.W_shar_mat[self.closest_neuron_idx],
                                        G.disconnected_grad(self.wr_relaxing_member)).sum()
        self.updates = sgd(self.cost_scal, [self.W_shar_mat], learning_rate=self.learning_rate)
        self.update_neurons = theano.function([self.x_row], self.cost_scal, updates=self.updates)


class ClusterRefiningSOM(WinnerRelaxingSOM):
    def __init__(self, weights, neurons_topology,
                 cr_adjusting_sigma=1.5, cr_adjusting_sigma_decay=0.9,
                 cr_learning_rate=0.005, cr_learning_rate_decay=0.9, **kwargs):
        super(ClusterRefiningSOM, self).__init__(weights, neurons_topology, **kwargs)

        self.cr_adjusting_sigma = theano.shared(cr_adjusting_sigma)
        self.cr_adjusting_sigma_decay = cr_adjusting_sigma_decay
        self.cr_learning_rate = theano.shared(cr_learning_rate)
        self.cr_learning_rate_decay = cr_learning_rate_decay

        self.affinities_to_data_point = T.exp(-self.distance_from_y_row/(self.cr_adjusting_sigma)**2)
        self.smoothed_distances_from_data_point = T.mul(self.distance_from_y_row, G.disconnected_grad(self.affinities_to_data_point))
        self.cr_affinity_cost_scal = self.smoothed_distances_from_data_point.sum()
        self.cr_updates = sgd(self.cr_affinity_cost_scal, [self.W_shar_mat], learning_rate=self.cr_learning_rate)
        self.cr_update_neurons = theano.function([self.x_row], self.cr_affinity_cost_scal, updates=self.cr_updates)

    def cr_phase_post_epoch(self, epoch):
        self.cr_adjusting_sigma.set_value(self.cr_adjusting_sigma.eval()*self.cr_adjusting_sigma_decay)
        self.cr_learning_rate.set_value(self.cr_learning_rate.eval()*self.cr_learning_rate_decay)

    def fit(self, X, number_of_epochs, with_history=HistoryLevel.NONE, cr_adjusting_epochs=0):
        if cr_adjusting_epochs:
            print("Original SOM fitting")
        super(WinnerRelaxingSOM, self).fit(X, number_of_epochs, with_history)
        if cr_adjusting_epochs:
            print("CR-phase")
            order = list(range(len(X)))
            for epoch in range(cr_adjusting_epochs):
                print("CR epoch {}".format(epoch+1))
                np.random.shuffle(order)
                for i in order:
                    self.cr_update_neurons(X[i])
                self.post_epoch(epoch)


class RectifyingSOM(ClusterRefiningSOM):
    def __init__(self, weights, neurons_topology, **kwargs):
        super(RectifyingSOM, self).__init__(weights, neurons_topology, **kwargs)
        self.excitement_score = theano.shared(np.zeros(shape=self.neurons_number))
        self.win_score = theano.shared(np.zeros(shape=self.neurons_number))

        self.excitement_accumulating_factor = theano.shared(1.0/self.learning_rate.eval())

        self.updates[self.win_score] = self.win_score + self.affinities_to_closest_neuron
        self.update_neurons = theano.function([self.x_row], self.cost_scal, updates=self.updates)

        self.excitement_updates_dict = OrderedDict()
        self.excitement_updates_dict[self.excitement_score] = \
            self.excitement_score + self.excitement_accumulating_factor*self.win_score
        self.update_excitement = theano.function([], self.excitement_score, updates=self.excitement_updates_dict)

    def post_epoch(self, epoch):
        BasicSOM.post_epoch(self, epoch)
        self.update_excitement()
        self.excitement_accumulating_factor.set_value(1.0/self.learning_rate.eval())
        self.win_score = theano.shared(np.zeros(shape=self.neurons_number))

    def get_processed_excitement(self):
        excitement = self.excitement_score.eval()
        mi, ma = np.min(excitement), np.max(excitement)
        excitement = 4*((excitement - mi)/(ma - mi)) - 1
        excitement[excitement > 1.0] = 1.0
        excitement[excitement < 0.0] = 0.0
        return excitement