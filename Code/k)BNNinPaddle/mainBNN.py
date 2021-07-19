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

from __future__ import print_function
import argparse
import ast
import numpy as np
from PIL import Image
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.optimizer import MissoOptimizer

from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable
import os

# for creating heterogeneous data loader
from het_reader import heterogenenous_reader
import itertools

import scipy.io

epsilon = 1e-6
lr = 0.001



delta = 0.01
W_dec = np.array([[1/3, 1/3, 0, 0, 1/3],
             [1/3, 1/3, 1/3, 0, 0],
             [0, 1/3, 1/3, 1/3, 0],
             [0, 0, 1/3, 1/3, 1/3],
             [1/3, 0, 0, 1/3, 1/3]]
             )


W_local = np.array([[1/5, 1/5, 1/5, 1/5, 1/5],
             [1/5, 1/5, 1/5, 1/5, 1/5],
             [1/5, 1/5, 1/5, 1/5, 1/5],
             [1/5, 1/5, 1/5, 1/5, 1/5],
             [1/5, 1/5, 1/5, 1/5, 1/5]]
             )

seed_data = 33
np.random.seed(seed_data)
center = []
for i in range(10):
    feature = np.random.normal(scale=5, size = (100))
    center.append(feature)

data_train_x = []
data_train_y = []
for i in range(10000):
    feature = np.random.normal(scale=2, size = (100))
    feature += center[int(i/1000)]
    feature = np.array(feature)
    data_train_x.append(feature/5)
  #  print(feature)
  #  print(feature.shape)
    data_train_y.append(int(i/1000))


data_test_x = []
data_test_y = []
for i in range(1000):
    feature = np.random.normal(scale=2, size = (100))
    feature += center[int(i/100)]
    feature = np.array(feature)
    data_test_x.append(feature/5)
  #  print(feature)
  #  print(feature.shape)
    data_test_y.append(int(i/100))




def parse_args():
    parser = argparse.ArgumentParser("Training for Mnist.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument("-e", "--epoch", default=5, type=int, help="set epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--ksync", default=1, type=int, help="k step average")
    parser.add_argument("--local", default=0, type=int, help="if it is local methods")


    parser.add_argument("--ce", action="store_true", help="run ce")
    parser.add_argument("-nw","--workers", default=5, type=int, help="number of worksers")
    args = parser.parse_args()
    return args


class ConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(ConvPool, self).__init__(name_scope)

        self._conv2d = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            self.full_name(),
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(MNIST, self).__init__(name_scope)

        self._fc_1 = FC(self.full_name(),
                      50,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=1/10)),
                      act="softmax")

        self._fc = FC(self.full_name(),
                      10,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=1/7)),
                      act="softmax")

    def forward(self, inputs, label=None):
        x = self._fc_1(inputs)
        x = self._fc(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

def test_mnist(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0].reshape(1, 100)
                              for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


def inference_mnist():
#    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
#        if args.use_data_parallel else fluid.CUDAPlace(0)
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        mnist_infer = MNIST("mnist")
        # load checkpoint
        model_dict, _ = fluid.dygraph.load_persistables("save_dir")
        mnist_infer.load_dict(model_dict)
        print("checkpoint loaded")

        # start evaluate mode
        mnist_infer.eval()

        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/infer_3.png')

        results = mnist_infer(to_variable(tensor_img))
        lab = np.argsort(results.numpy())
        print("Inference result of image/infer_3.png is: %d" % lab[0][-1])



def train_mnist(args):
    print('path is')
    print(os.getcwd())

    epoch_num = args.epoch
    n_workers = args.workers
    k_period = args.ksync
    BATCH_SIZE = 64
    beta2 = 0.99
    trainer_count = fluid.dygraph.parallel.Env().nranks
 #   epsilon = 0.1  # initial value of moment2 in DAMSGrad
    avg_loss_sav = []
    
    test_loss_sav = []
    
    test_acc_sav = []

    if args.local == 1:
        W = W_local
    else:
        W = W_dec
 


    SGDM = False # use SGDM or DAMSGrad
  #  place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
  #      if args.use_data_parallel else fluid.CUDAPlace(0)
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            
        dist_models = []
        optimizers = []
        adaptive_learning_rates = []
        avg_dic_moment2_max = []
        adaptive_learning_rates_max = []
        alg_u = []
        alg_tilde_u = []

# creat heterogeneous data loader
        readers = []
        mnist_reader = paddle.dataset.mnist.train()
        images = data_train_x
        labels = data_train_y
     #   for item in mnist_reader():
    #        images.append(item[0])
   #         labels.append(item[1])
        idx = np.argsort(labels)

        y_train_sorted = [labels[i] for i in idx]
        x_train_sorted = [images[i] for i in idx]

    # shuffle local data
        n_batch = int(10000/BATCH_SIZE)
        batch_per_worker = int(n_batch/n_workers)
        sample_per_worker = batch_per_worker * BATCH_SIZE        
        for i in range(n_workers):
            permute = np.random.permutation(sample_per_worker)
            data_slice = y_train_sorted[i*sample_per_worker : (i+1)*sample_per_worker]
            data_slice = [data_slice[permute[i]] for i in range(sample_per_worker)]
            y_train_sorted[i*sample_per_worker : (i+1)*sample_per_worker] = data_slice

            data_slice = x_train_sorted[i*sample_per_worker : (i+1)*sample_per_worker]
            data_slice = [data_slice[permute[i]] for i in range(sample_per_worker)]
            x_train_sorted[i*sample_per_worker : (i+1)*sample_per_worker] = data_slice


        def seq_reader(x_train, y_train):
            def reader():
                for item in zip(x_train, y_train):
                    yield item[0], item[1]
            return reader

        mnist_seq_reader = seq_reader(x_train_sorted, y_train_sorted)
        def create_reader():
            return mnist_seq_reader



        for i in range(n_workers):
            mnist = MNIST("mnist"+str(i))
            optimizer = MissoOptimizer(learning_rate=args.lr)
            dist_models.append(mnist)
            optimizers.append(optimizer)
            adaptive_learning_rates.append({})
            adaptive_learning_rates_max.append({})
            alg_u.append({})
            alg_tilde_u.append({})
            avg_dic_moment2_max.append({})
            print(mnist.parameters())
            moment2 = {}
            
            #homogeneous data loader
    #        train_reader = paddle.batch(
    #            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)

            # heterogeneous data loader
            train_reader = paddle.batch(
                seq_reader(x_train_sorted, y_train_sorted), batch_size=BATCH_SIZE, drop_last=True) 
            test_reader_2 = paddle.batch(
                seq_reader(data_test_x, data_test_y), batch_size=1000, drop_last=True)

            hetero_reader = heterogenenous_reader(train_reader, i, n_workers, int(10000/BATCH_SIZE))
            readers.append(hetero_reader)

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)
        itr = 0
        for epoch in range(epoch_num):
            #initialize data generators
            readers_gen = [reader() for reader in readers]
            readers_zip = zip(*readers_gen)
            interleave_reader = itertools.chain.from_iterable(readers_zip)

            for batch_id, data in enumerate(interleave_reader):                
                id_worker = itr % n_workers
#                
              #  print(itr)
                
                dy_x_data = np.array([x[0].reshape(1, 100)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)
                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True
# average models and update adaptive learning rates
                if itr % (n_workers*k_period) == 0:
                    model_main = dist_models[0]
                    cost, acc = model_main(img, label)   
#                    avg = fluid.average.WeightedAverage()
#                    scope = fluid.global_scope();
#                    print('printing all variable')
#                    print(list(fluid.default_main_program().list_vars()))
# initialize the dictionary of moment2 and initialize all models
                    if itr == 0:  
                        for i in range(n_workers):
                            cost, acc = dist_models[i](img, label)
                                                     
                            if not adaptive_learning_rates[i]:
                                moment2_dic = {}
                                moment2_max_dic = {}
                                alg_tilde_u_dic = {}
                                alg_u_dic = {}
                                
                                for param in dist_models[i].parameters():
                                    print(param.name)
                                    param_array = param.numpy() # to get size of parameters
                                    
                                    moment2_param = epsilon*np.ones(param_array.shape)
                                    moment2_param_var = to_variable(moment2_param.astype(np.float32))
                                    moment2_param_var.stop_gradient = True
                                    moment2_param_var.persistable = True
                                    moment2_dic[param.name] = moment2_param_var
                                    
                                    moment2_max_param = epsilon*np.ones(param_array.shape)
                                    moment2_max_param_var = to_variable(moment2_max_param.astype(np.float32))
                                    moment2_max_param_var.stop_gradient = True
                                    moment2_max_param_var.persistable = True
                                    moment2_max_dic[param.name] = moment2_max_param_var    
                                    
                                    alg_tilde_u_param = moment2_max_param
                                    alg_tilde_u_param_var = to_variable(alg_tilde_u_param.astype(np.float32))
                                    alg_tilde_u_param_var.stop_gradient = True
                                    alg_tilde_u_param_var.persistable = True
                                    alg_tilde_u_dic[param.name] = alg_tilde_u_param_var
                                    
                                    alg_u_param = max(epsilon, delta)*np.ones(param_array.shape)
                                    alg_u_param_var = to_variable(alg_u_param.astype(np.float32))
                                    alg_u_param_var.stop_gradient = True
                                    alg_u_param_var.persistable = True
                                    alg_u_dic[param.name] = alg_u_param_var    
                                      
                                adaptive_learning_rates[i] = moment2_dic
                                adaptive_learning_rates_max[i] = moment2_max_dic
                                alg_tilde_u[i] = alg_tilde_u_dic
                                alg_u[i] = alg_u_dic

                                    

# calculate averated parameters and averaged moment2
                    avg_dict = []
#                    avg_dic_moment2 = []
                    avg_dic_tilde_u = []
                    for i in range(n_workers):
                        avg_dict.append({})
#                        avg_dic_moment2.append({})
                        avg_dic_tilde_u.append({})
#                    avg_dict = {}
#                    avg_dic_moment2 = {}
                    
                        for param in model_main.parameters():
                            avg_dict[i][param.name[6:]] = fluid.average.WeightedAverage()
#                            avg_dic_moment2[i][param.name[6:]] = fluid.average.WeightedAverage()
                            avg_dic_tilde_u[i][param.name[6:]] = fluid.average.WeightedAverage()
                    for i in range(n_workers):
                        for j in range(n_workers):
#                            moment2_dic = adaptive_learning_rates[i]
                            alg_tilde_u_dic = alg_tilde_u[j]
                            for param in dist_models[j].parameters():
                                # compute weighted average of parameters
                                avg_dict[i][param.name[6:]].add(param.numpy(),W[i,j])
                                # compute weighted average of tilde_u
                                avg_dic_tilde_u[i][param.name[6:]].add(alg_tilde_u_dic[param.name].numpy(),W[i,j])
#                                print(param.name)
#                                print(adaptive_learning_rates_max[j][param.name].numpy())
#                                avg_dic_moment2[i][param.name[6:]].add(moment2_dic[param.name].numpy(),1)
                        for param in dist_models[i].parameters():
                            alg_u[i][param.name]._ivar.value().get_tensor().set(np.maximum(avg_dic_tilde_u[i][param.name[6:]].eval(), delta),
                                       fluid.CPUPlace())
    # initialize max of moment2 at the first iteration
#                    if itr == 0:
#                        for i in range(n_workers):
#                            for name, avg_moment2 in avg_dic_moment2[i].items():
#                                avg_dic_moment2_max[i][name] = to_variable(avg_moment2.eval())
#                                avg_dic_moment2_max[i][name].persistable = True
                    #        print(param.name)
    # update the max parameters
#                    for i in range(n_workers):
#                        for name, avg_moment2 in avg_dic_moment2[i].items():
#                            val_max_moment2 = avg_dic_moment2_max[i][name].numpy()
#                            val_max_moment2 = np.maximum(val_max_moment2,avg_moment2.eval())
#                            avg_dic_moment2_max[i][name]._ivar.value().get_tensor().set(val_max_moment2,
#                                       fluid.CPUPlace())

    
    
                #    name_parameters = list(avg_dict)
                #    for name in name_parameters:
    # average the parameters of all models
                    for i in range(n_workers):
                        for param in dist_models[i].parameters():

                            var = param._ivar.value()
                            tensor = var.get_tensor()
                            tensor.set(avg_dict[i][param.name[6:]].eval(),
                                       fluid.CPUPlace())
#                            param.get_tensor().set(avg_dict[param.name[6:]].eval(),fluid.CPUPlace())
#                            print(param.name)
#                            a = scope.find_var('mnist'+str(i)+name)
#                            print(a)
#                            scope.find_var('mnist'+str(i)+name).get_tensor().set(avg_dict[name].eval(),fluid.CPUPlace())
            # switch to ith worker
                mnist = dist_models[id_worker]
            #    print(mnist.parameters())
                optimizer = optimizers[id_worker]
                
                cost, acc = mnist(img, label)     
                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
            #    
            #    scope = fluid.global_scope();
            #    scope.list_vars()
                itr = itr + 1
                



#                if args.use_data_parallel:
#                    avg_loss = mnist.scale_loss(avg_loss)
#                    avg_loss.backward()
#                    mnist.apply_collective_grads()
#                else:


                avg_loss.backward()

                
                params_grads = optimizer.backward(avg_loss, parameter_list = mnist.parameters())
                var_list = []
                moment2 = []
                # update individual moment2
                for param_and_grad in params_grads:
                    var_list.append(param_and_grad[0])
                    #print(param_and_grad[0].name)
                      #  print(adaptive_learning_rates[id_worker])

                    moment2_param = adaptive_learning_rates[id_worker][param_and_grad[0].name]
                    moment2_max_param = adaptive_learning_rates_max[id_worker][param_and_grad[0].name]
                    alg_tilde_u_param = alg_tilde_u[id_worker][param_and_grad[0].name]
                    alg_u_param = alg_u[id_worker][param_and_grad[0].name]

                    if not SGDM:
                        moment2_val = (beta2) * moment2_param.numpy() + (1-beta2) * param_and_grad[1].numpy()**2
                        moment2_max_pre_val = moment2_max_param.numpy()
                        moment2_max_val = np.maximum(moment2_max_param.numpy(), moment2_val)
                        moment2_param._ivar.value().get_tensor().set(moment2_val, fluid.CPUPlace())
                        moment2_max_param._ivar.value().get_tensor().set(moment2_max_val, fluid.CPUPlace())
                        alg_tilde_u_param._ivar.value().get_tensor().set(alg_tilde_u_param.numpy() + moment2_max_val - moment2_max_pre_val, fluid.CPUPlace())
#                    moment2_val = (beta2) * moment2_param.numpy() + (1-beta2) * param_and_grad[1].numpy()**2
#                    moment2_param._ivar.value().get_tensor().set(moment2_val, fluid.CPUPlace())
#                    moment2_param = (beta2)*moment2_param + (1-beta2)*fluid.layers.square(param_and_grad[1])+ 0.001
#                    adaptive_learning_rates[id_worker][param_and_grad[0].name] = moment2_param
#
#                    print(param_and_grad[0].name)
#                    param_array = param_and_grad[0].numpy()
#                    grad = param_and_grad[0].numpy()
#                    print(grad)
#                    moment2_param = np.ones(param_array.shape)
#                    moment2_param.astype(np.float32)
#                    moment2_param_var = to_variable(moment2_param.astype(np.float32))
#                    moment2_param_var.stop_gradient = True
#                    moment2.append(moment2_param)
#                    moment2.append(avg_dic_moment2_max[param_and_grad[0].name[6:]])
                    moment2.append(moment2_max_param)
#                print(len(moment2))
#                print('params backwarded')
#
#                for param in mnist.parameters():
#                    print(param.name)
#                    param_array = param.numpy()
#              #      grad = param_and_grad[0].numpy()
#              #      print(grad)
#                    moment2_param = np.ones(param_array.shape)
#                #    moment2_param.astype(np.float32)
#                    moment2_param_var = to_variable(moment2_param.astype(np.float32))
#                    moment2_param_var.stop_gradient = True
#                    moment2.append(moment2_param_var)
#                var_list = []
#                for param in mnist.parameters():
#                    var_list.append(param)
#                    print(param.name)
#                print(var_list)
#                    , parameter_list = mnist.parameters()
                
                optimizer.minimize(avg_loss, parameter_list = var_list, moment2 = moment2)
                mnist.clear_gradients()
                if batch_id % 60 == 0:
                    print("Loss at epoch {} step {}: {:}".format(
                        epoch, batch_id, avg_loss.numpy()))
                
                avg_loss_sav.append(avg_loss.numpy())
                
                if itr % 250 == 0:
                    mnist.eval()
                    test_cost, test_acc = test_mnist(test_reader_2, mnist, 1000)
                    test_loss_sav.append(test_cost)
                    test_acc_sav.append(test_acc)
                    mnist.train()
                
  ####### # added another model to train       
            #    mnist2 = MNIST("mnist2")
            #    mnist2.clear_gradients()
#                cost2, acc2 = mnist2(img, label)
#                loss2 = fluid.layers.cross_entropy(cost2, label)
#                avg_loss2 = fluid.layers.mean(loss2)
#                moment2 = []
#                for param in mnist2.parameters():
#                   # print(param.name)
#                    param_array = param.numpy()
#              #      grad = param_and_grad[0].numpy()
#              #      print(grad)
#                    moment2_param = np.ones(param_array.shape)
#                #    moment2_param.astype(np.float32)
#                    moment2_param_var = to_variable(moment2_param.astype(np.float32))
#                    moment2_param_var.stop_gradient = True
#                    moment2.append(moment2_param_var)
#                avg_loss2.backward()
#                params_grads = optimizer2.backward(avg_loss2)
#                optimizer2.minimize(avg_loss2, parameter_list = mnist2.parameters(), moment2 = moment2)
#                mnist2.clear_gradients()
#                # save checkpoint
#                if batch_id % 100 == 0:
#                    print("Loss at epoch {} step {}: {:}".format(
#                        epoch, batch_id, avg_loss2.numpy()))
########
  
  
            mnist.eval()
            test_cost, test_acc = test_mnist(test_reader_2, mnist, 1000)
        # switch to train mode
            mnist.train()
            
            
            
            if args.ce:
                print("kpis\ttest_acc\t%s" % test_acc)
                print("kpis\ttest_cost\t%s" % test_cost)
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                epoch, test_cost, test_acc))

        fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir")
        print("checkpoint saved")
        if args.local == 1:
            scipy.io.savemat('local_adam_het/adam_s'+str(args.lr)+'.mat', {'avg_loss_adam'+str(args.lr).replace('.', ''):np.array(avg_loss_sav),
                                      'test_loss_adam'+str(args.lr).replace('.', ''): np.array(test_loss_sav),
                                      'test_acc_adam'+str(args.lr).replace('.', ''): np.array(test_acc_sav)})
        else:
        
            scipy.io.savemat('dadam_het/adam_s'+str(args.lr)+'.mat', {'avg_loss_adam'+str(args.lr).replace('.', ''):np.array(avg_loss_sav),
                                      'test_loss_adam'+str(args.lr).replace('.', ''): np.array(test_loss_sav),
                                      'test_acc_adam'+str(args.lr).replace('.', ''): np.array(test_acc_sav)})


     #   inference_mnist()


if __name__ == '__main__':
    args = parse_args()
    train_mnist(args)
