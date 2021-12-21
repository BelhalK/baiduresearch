# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle

import paddle.fluid as fluid

from bayesian import distributions

class RandomWeights(object):
    """
    
    """

    def __init__(self, bn, name, dist, observation=None, **kwargs):
        if bn is None:
            pass
        self._bn = bn

        self._name = name
        self._dist = dist
        self._dtype = dist.dtype
        #print(kwargs)
        self._n_samples = kwargs.get("n_samples", None)
        self._observation = observation
        super(RandomWeights, self).__init__()

        ## when computing log_prob, which dims are averaged or summed
        self._reduce_mean_dims = kwargs.get("reduce_mean_dims", None)
        self._reduce_sum_dims = kwargs.get("reduce_sum_dims", None)
        self._multiplier = kwargs.get("multiplier", None)

    def _check_observation(self, observation):
        return observation

    @property
    def bn(self):
        """
        The :class:`BayesianNeuralNet` where the :class:`RandomWeights` lives.
        :return: A :class:`BayesianNeuralNet` instance.
        """
        return self._bn

    @property
    def name(self):
        """
        The name of the :class:`RandomWeights`.
        :return: A string.
        """
        return self._name

    @property
    def dtype(self):
        """
        The sample type of the :class:`RandomWeights`.
        :return: A ``DType`` instance.
        """
        return self._dtype

    @property
    def dist(self):
        """
         The distribution followed by the :class:`RandomWeights`.
        :return: A :class:`~bayesian.distributions.base.Distribution` instance.
        """
        return self._dist

    def is_observed(self):
        """
        Whether the :class:`RandomWeights` is observed or not.
        :return: A bool.
        """
        return self._observation is not None

    @property
    def tensor(self):
        """
        The value of this :class:`RandomWeights`. If it is observed, then
        the observation is returned, otherwise samples are returned.
        :return: A Tensor.
        """
        if self._name in self._bn.observed.keys():
            self._dist.sample_cache = self._bn.observed[self._name]
            return self._bn.observed[self._name]
        else:
            _samples = self._dist.sample(n_samples=self._n_samples)
        return _samples

    @property
    def shape(self):
        """
        Return the static shape of this :class:`RandomWeights`.
        :return: A ``TensorShape`` instance.
        """
        return self.tensor.shape

    def log_prob(self,sample=None):
        _log_probs = self._dist.log_prob(sample)

        if self._reduce_mean_dims:
            _log_probs = fluid.layers.reduce_mean(_log_probs, self._reduce_mean_dims, keep_dim=True)

        if self._reduce_sum_dims:
            _log_probs = fluid.layers.reduce_sum(_log_probs, self._reduce_sum_dims, keep_dim=True)

        if self._reduce_mean_dims or self._reduce_sum_dims:
            _m = self._reduce_mean_dims if self._reduce_mean_dims else []
            _s = self._reduce_sum_dims if self._reduce_sum_dims else []
            _log_probs = fluid.layers.squeeze(_log_probs, [*_m, *_s])

        if self._multiplier:
            _log_probs = _log_probs * self._multiplier

        return _log_probs