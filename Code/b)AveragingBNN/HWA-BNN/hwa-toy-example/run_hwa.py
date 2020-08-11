#python run_hwa.py --optim hwa --nb_points 500 --lr 0.08 --start_avg 10 --avg_period 100

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

import warnings
warnings.filterwarnings('ignore')

from keras.layers import Input
from keras.models import Model
from keras import callbacks, optimizers

from options import args_parser
from hwa import HWA
# import tqdm

args = args_parser()

def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon

train_size = 32
noise = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
y = f(X, sigma=noise)
y_true = f(X, sigma=0.0)

plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_true, label='Truth')
plt.title('Noisy training data and ground truth')
plt.legend()
# plt.show()




class DenseVariational(Layer):
    def __init__(self,
                 args,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.args = args
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.n_models = 0
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',#weight mean
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                         trainable=True)
        # self.kernel_mu_hwa = self.kernel_mu
        # self.bias_mu_hwa = self.bias_mu
        self.bias_mu = self.add_weight(name='bias_mu',#bias mean
                                       shape=(self.units,),
                                       initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', #weight mean
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', #bias variance
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        # kernel_mu_hwa = self.kernel_mu_hwa
        # bias_mu_hwa = self.bias_mu
        # self.n_models = self.args.epochs #update nmodels for averaging
        # kernel_mu_hwa = (self.n_models*kernel_mu_hwa + mu)/(self.n_models+1) #HWA mean
        # variational_dist = tfp.distributions.Normal(kernel_mu_hwa, sigma)
        
        variational_dist = tfp.distributions.Normal(mu, sigma)

        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))



batch_size = train_size
num_batches = train_size / batch_size

kl_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 1.5, 
    'prior_sigma_2': 0.1, 
    'prior_pi': 0.5 
}

x_in = Input(shape=(1,))
x = DenseVariational(args, 20, kl_weight, **prior_params, activation='relu')(x_in)
x = DenseVariational(args, 20, kl_weight, **prior_params, activation='relu')(x)
x = DenseVariational(args, 1, kl_weight, **prior_params)(x)

model = Model(x_in, x)

def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_obs))

if args.optim == 'adam':
    opt = optimizers.Adam(lr=args.lr) #0.08 default
elif args.optim == 'sgd':
    opt = optimizers.SGD(lr=args.lr)
elif args.optim == 'rmsprop':
    opt = optimizers.RMSprop(lr=args.lr)
elif args.optim == 'hwa':
    # opt = optimizers.SGD(lr=args.lr)
    opt = optimizers.Adam(lr=args.lr)
    opt = tfa.optimizers.SWA(opt, start_averaging=args.start_avg, average_period=args.avg_period)
    # opt = HWA(lr=args.lr)

model.compile(loss=neg_log_likelihood, optimizer=opt, metrics=['mse'])
model.fit(X, y, batch_size=batch_size, epochs=args.epochs, verbose=0);

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []


#Plot uncertainty
# for i in tqdm.tqdm(range(500)):
for i in range(args.nb_points):
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)
    if i%100 == 0:
        print(i)
    

y_preds = np.concatenate(y_pred_list, axis=1)
y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1) #standard deviation of the predictions --> area of the epistemic uncertainty
plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(), y_mean + 2 * y_sigma, 
                        y_mean - 2 * y_sigma, 
                        alpha=0.5, label='Epistemic uncertainty')
plt.title('Prediction')
plt.legend()
plt.savefig('./save/opt_{}.png'.format(args.optim))
# plt.show()