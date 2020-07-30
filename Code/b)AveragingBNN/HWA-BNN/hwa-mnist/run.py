# python3 run.py --num_epochs 2 --viz_steps 200 --num_monte_carlo 5 --optimizer adam
# python3 run.py --num_epochs 2 --viz_steps 200 --num_monte_carlo 5 --optimizer sgd --learning_rate 0.01
# python3 run.py --num_epochs 2 --viz_steps 200 --num_monte_carlo 5 --optimizer hwa --start_avg 10 --avg_period 100
# python3 run.py --num_epochs 2 --viz_steps 200 --num_monte_carlo 5 --optimizer hwa_sgd --start_avg 10 --avg_period 10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from absl import app
from absl import flags
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from logger import Logger, savefig
import optimizers as optim_local

tf.enable_v2_behavior()

import pdb
warnings.simplefilter(action='ignore')

try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]
NUM_TRAIN_EXAMPLES = 60000
NUM_HELDOUT_EXAMPLES = 10000
NUM_CLASSES = 10

flags.DEFINE_float('learning_rate',
                   default=0.01,
                   help='Initial learning rate.')
flags.DEFINE_integer('num_epochs',
                     default=10,
                     help='Number of training steps to run.')
flags.DEFINE_integer('batch_size',
                     default=128,
                     help='Batch size.')
flags.DEFINE_string('data_dir',
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'),
                    help='Directory where data is stored (if using real data).')
flags.DEFINE_string('model_dir',default=os.path.join(os.getcwd(),'bayesian_neural_network/'),help="Directory to put the model's fit.")
flags.DEFINE_integer('viz_steps',
                     default=400,
                     help='Frequency at which save visualizations.')
flags.DEFINE_integer('num_monte_carlo',
                     default=50,
                     help='Network draws to compute predictive probabilities.')
flags.DEFINE_bool('fake_data',
                  default=False,
                  help='If true, uses fake data. Defaults to real data.')
flags.DEFINE_string('optimizer', 
                  default='sgd',help='optimizer to use (sgd, adam)')
#HWA params
flags.DEFINE_integer('start_avg',default=10,help='Start Averaging in HWA')
flags.DEFINE_integer('avg_period',default=10,help='Averaging Period in HWA')

FLAGS = flags.FLAGS



def create_model():
  # KL divergence weighted by the number of training samples, using
  # lambda function to pass as input to the kernel_divergence_fn on flipout layers.
  kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

  model = tf.keras.models.Sequential([
      tfp.layers.Convolution2DFlipout(
          6, kernel_size=5, padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(
          pool_size=[2, 2], strides=[2, 2],
          padding='SAME'),
      tfp.layers.Convolution2DFlipout(
          16, kernel_size=5, padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(
          pool_size=[2, 2], strides=[2, 2],
          padding='SAME'),
      tfp.layers.Convolution2DFlipout(
          120, kernel_size=5, padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tf.keras.layers.Flatten(),
      tfp.layers.DenseFlipout(
          84, kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tfp.layers.DenseFlipout(
          NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.softmax)
  ])

  # Model compilation.
  # pdb.set_trace()
  if FLAGS.optimizer =='adam':
      optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
  elif FLAGS.optimizer == 'sgd':
      optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate)
  elif FLAGS.optimizer == 'sgld':
      optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(FLAGS.learning_rate, preconditioner_decay_rate=0.95, data_size=1, burnin=25,
    diagonal_bias=1e-08, name=None, parallel_iterations=10)
  elif FLAGS.optimizer == 'hwa':
      optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
      optimizer = tfa.optimizers.SWA(optimizer, start_averaging=FLAGS.start_avg, average_period=FLAGS.avg_period)
  elif FLAGS.optimizer == 'hwa_sgd':
      optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate)
      optimizer = tfa.optimizers.SWA(optimizer, start_averaging=FLAGS.start_avg, average_period=FLAGS.avg_period)


  # We use the categorical_crossentropy loss since the MNIST dataset contains
  # ten labels. The Keras API will then automatically add the
  # Kullback-Leibler divergence (contained on the individual layers of the model), to the cross entropy loss, effectively calcuating the (negated) Evidence Lower Bound Loss (ELBO)
  model.compile(optimizer, loss='categorical_crossentropy',
                metrics=['accuracy'], experimental_run_tf_function=False)
  return model


class MNISTSequence(tf.keras.utils.Sequence):
  """Produces a sequence of MNIST digits with labels."""

  def __init__(self, data=None, batch_size=128, fake_data_size=None):
    """Initializes the sequence.
    Args:
      data: Tuple of numpy `array` instances, the first representing images and
            the second labels.
      batch_size: Integer, number of elements in each training batch.
      fake_data_size: Optional integer number of fake datapoints to generate.
    """
    if data:
      images, labels = data
    else:
      images, labels = MNISTSequence.__generate_fake_data(
          num_images=fake_data_size, num_classes=NUM_CLASSES)
    self.images, self.labels = MNISTSequence.__preprocessing(
        images, labels)
    self.batch_size = batch_size

  @staticmethod
  def __generate_fake_data(num_images, num_classes):
    images = np.random.randint(low=0, high=256,
                               size=(num_images, IMAGE_SHAPE[0],
                                     IMAGE_SHAPE[1]))
    labels = np.random.randint(low=0, high=num_classes,
                               size=num_images)
    return images, labels

  @staticmethod
  def __preprocessing(images, labels):
    """Preprocesses image and labels data."""
    images = 2 * (images / 255.) - 1.
    images = images[..., tf.newaxis]

    labels = tf.keras.utils.to_categorical(labels)
    return images, labels

  def __len__(self):
    return int(tf.math.ceil(len(self.images) / self.batch_size))

  def __getitem__(self, idx):
    batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
    return batch_x, batch_y


def main(argv):
  del argv  # unused
  if tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warning(
        'Warning: deleting old log directory at {}'.format(FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)
  
  print('==> Preparing data..')
  if FLAGS.fake_data:
    train_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                              fake_data_size=NUM_TRAIN_EXAMPLES)
    heldout_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                                fake_data_size=NUM_HELDOUT_EXAMPLES)
  else:
    train_set, heldout_set = tf.keras.datasets.mnist.load_data()
    train_seq = MNISTSequence(data=train_set, batch_size=FLAGS.batch_size)
    heldout_seq = MNISTSequence(data=heldout_set, batch_size=FLAGS.batch_size)
  
  print('==> Building Model..')
  model = create_model()
  model.build(input_shape=[None, 28, 28, 1])

  
  print('==> Creating Logger Files..')
  #Create checkpoints log file
  logname = 'LeNet5'
  dataset = 'MNIST'
  title = '{}-{}'.format(dataset, logname)
  checkpoint_dir = 'checkpoint/checkpoint_{}'.format(dataset)
  loggertrain = Logger('{}/trainlog{}_opt{}_lr{}_bs{}_avgp{}.txt'.format(checkpoint_dir, logname,  FLAGS.optimizer, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.avg_period), title=title)
  loggertrain.set_names(['Learning Rate', 'Train Loss','Train Acc.'])

  loggertest = Logger('{}/testlog{}_opt{}_lr{}_bs{}_avgp{}.txt'.format(checkpoint_dir, logname,  FLAGS.optimizer, FLAGS.learning_rate, FLAGS.batch_size,FLAGS.avg_period), title=title)
  loggertest.set_names(['Learning Rate', 'Test Loss',  'Test Acc.'])

  print('==> Training Phase...')
  for epoch in range(FLAGS.num_epochs):
    epoch_accuracy, epoch_loss = [], []
    for step, (batch_x, batch_y) in enumerate(train_seq):
      batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
      epoch_accuracy.append(batch_accuracy)
      epoch_loss.append(batch_loss)
      
      #write in the logger ['Learning Rate', 'Train Loss', 'Train Acc.']
      loggertrain.append([FLAGS.learning_rate, batch_loss, batch_accuracy])

      if step % 100 == 0:
        print('Epoch: {}, Batch index: {}, '
              'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                  epoch, step,
                  tf.reduce_mean(epoch_loss),
                  tf.reduce_mean(epoch_accuracy)))
    print('==> Testing Phase...')
    #testing              
    for step, (batch_x, batch_y) in enumerate(heldout_seq):
      test_loss, test_accuracy = model.train_on_batch(batch_x, batch_y)      
      #write in the testlogger ['Learning Rate','Test Loss','Test Acc.']
      loggertest.append([FLAGS.learning_rate, test_loss, test_accuracy])

    #   if (step+1) % FLAGS.viz_steps == 0:
    #     # Compute log prob of heldout set by averaging draws from the model:
    #     # p(heldout | train) = int_model p(heldout|model) p(model|train)
    #     #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
    #     # where model_i is a draw from the posterior p(model|train).
    #     print(' ... Running monte carlo inference')
    #     probs = tf.stack([model.predict(heldout_seq, verbose=1)
    #                       for _ in range(FLAGS.num_monte_carlo)], axis=0)
    #     mean_probs = tf.reduce_mean(probs, axis=0)
    #     heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
    #     print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

    #     if HAS_SEABORN:
    #       names = [layer.name for layer in model.layers
    #                if 'flipout' in layer.name]
    #       qm_vals = [layer.kernel_posterior.mean()
    #                  for layer in model.layers
    #                  if 'flipout' in layer.name]
    #       qs_vals = [layer.kernel_posterior.stddev()
    #                  for layer in model.layers
    #                  if 'flipout' in layer.name]
    #       plot_weight_posteriors(names, qm_vals, qs_vals,
    #                              fname=os.path.join(
    #                                  FLAGS.model_dir,
    #                                  'epoch{}_step{:05d}_weights.png'.format(
    #                                      epoch, step)))
    #       plot_heldout_prediction(heldout_seq.images, probs,
    #                               fname=os.path.join(
    #                                   FLAGS.model_dir,
    #                                   'epoch{}_step{}_pred.png'.format(
    #                                       epoch, step)),
    #                               title='mean heldout logprob {:.2f}'
    #                               .format(heldout_log_prob))


if __name__ == '__main__':
  app.run(main)





def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
  ax.set_title('weight means')
  ax.set_xlim([-1.5, 1.5])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
  ax.set_title('weight stddevs')
  ax.set_xlim([0, 1.])

  fig.tight_layout()
  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


def plot_heldout_prediction(input_vals, probs,fname, n=10, title=''):
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation='None')

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title('posterior samples')

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), tf.reduce_mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('predictive probs')
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))
