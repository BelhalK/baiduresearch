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

import paddle
import paddle.fluid as fluid

def log_mean_exp(x, dim=None, keepdims=False):
    """
    Tensorflow numerically stable log mean of exps across the `dim`.
    :param x: A Tensor.
    :param dim: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keepdims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.
    :return: A Tensor after the computation of log mean exp along given axes of
        x.
    """
    x_max = fluid.layers.reduce_max(x, dim=dim, keep_dim=True)
    ret = paddle.log(fluid.layers.reduce_mean(paddle.exp(x - x_max), dim=dim,
                                keep_dim=True)) + x_max
    if not keepdims:
        ret = fluid.layers.reduce_mean(ret, dim=dim)
    return ret