#python3 final_runsCIFAR_adam.py --batchsize=128 --nbepochs=30 --nbruns=4
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import warnings
import os


# Dependency imports
import argparse
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import pickle


from models.bayesian_resnet import bayesian_resnet
from models.bayesian_vgg import bayesian_vgg

# warnings.simplefilter(action="ignore")
tfd = tfp.distributions

IMAGE_SHAPE = [32, 32, 3]


ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batchsize", type=int, default=1,help="")
ap.add_argument("-e", "--nbepochs", type=int, default=1,help="")
ap.add_argument("-r", "--nbruns", type=int, default=1,help="")
args = vars(ap.parse_args())

def build_input_pipeline(x_train, x_test, y_train, y_test,
                         batch_size, valid_size):
  """Build an Iterator switching between train and heldout data."""
  x_train = x_train.astype("float32")
  x_test = x_test.astype("float32")

  x_train /= 255
  x_test /= 255

  y_train = y_train.flatten()
  y_test = y_test.flatten()

  if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

  print("x_train shape:" + str(x_train.shape))
  print(str(x_train.shape[0]) + " train samples")
  print(str(x_test.shape[0]) + " test samples")

  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, np.int32(y_train)))
  training_batches = training_dataset.shuffle(
      50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (x_test, np.int32(y_test)))
  heldout_batches = heldout_dataset.repeat().batch(valid_size)
  heldout_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_batches)

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.compat.v1.placeholder(tf.string, shape=[])
  feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator


def build_fake_data(num_examples):
  num_examples = num_examples
  x_train = np.random.rand(num_examples, *IMAGE_SHAPE).astype(np.float32)
  y_train = np.random.permutation(np.arange(num_examples)).astype(np.int32)
  x_test = np.random.rand(num_examples, *IMAGE_SHAPE).astype(np.float32)
  y_test = np.random.permutation(np.arange(num_examples)).astype(np.int32)
  return (x_train, y_train), (x_test, y_test)

model_dir = "bnnmodels/"


def run_experiment(algo,fake_data, batch_size, epochs, learning_rate,verbose):
    with tf.Session() as sess:
        
        model_fn = bayesian_resnet
        #model_fn = bayesian_vgg
        model = model_fn(
            IMAGE_SHAPE,
            num_classes=10,
            kernel_posterior_scale_mean=kernel_posterior_scale_mean,
            kernel_posterior_scale_constraint=kernel_posterior_scale_constraint)
        logits = model(images)
        labels_distribution = tfd.Categorical(logits=logits)
        t = tf.compat.v2.Variable(0.0)
        kl_regularizer = t / (kl_annealing * len(x_train) / batch_size)

        log_likelihood = labels_distribution.log_prob(labels)
        neg_log_likelihood = -tf.reduce_mean(input_tensor=log_likelihood)
        kl = sum(model.losses) / len(x_train) * tf.minimum(1.0, kl_regularizer)
        loss = neg_log_likelihood + kl

        predictions = tf.argmax(input=logits, axis=1)

        with tf.compat.v1.name_scope("train"):
            train_accuracy, train_accuracy_update_op = tf.compat.v1.metrics.accuracy(
              labels=labels, predictions=predictions)
            if algo=="adam":
                opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            if algo=="adagrad":
                opt = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate)
            if algo=="adadelta":
                opt = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=learning_rate)
            if algo=="rmsprop":
                opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)

            train_op = opt.minimize(loss)
            update_step_op = tf.compat.v1.assign(t, t + 1)

        with tf.compat.v1.name_scope("valid"):
            valid_accuracy, valid_accuracy_update_op = tf.compat.v1.metrics.accuracy(
              labels=labels, predictions=predictions)

            init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                           tf.compat.v1.local_variables_initializer())

            stream_vars_valid = [v for v in tf.compat.v1.local_variables() if "valid/" in v.name]
            reset_valid_op = tf.compat.v1.variables_initializer(stream_vars_valid)
    
    
   # with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        # Run the training loop
        train_handle = sess.run(training_iterator.string_handle())
        heldout_handle = sess.run(heldout_iterator.string_handle())
        training_steps = int(
          round(epochs * (len(x_train) / batch_size)))
        
        listkl = []
        listloss = []
        listaccuracy = []
        print(training_steps)
        for step in range(training_steps):
            _ = sess.run([train_op,
                      train_accuracy_update_op,
                      update_step_op],
                     feed_dict={handle: train_handle})
            # Print loss values
            #set_trace()
            loss_value, accuracy_value, kl_value = sess.run(
                  [loss, train_accuracy, kl], feed_dict={handle: train_handle})
            if step % 100 == 0:
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f} KL: {:.3f}".format(
                      step, loss_value, accuracy_value, kl_value))
            listkl.append(kl_value)
            listloss.append(loss_value)
            listaccuracy.append(accuracy_value)
        sess.run(reset_valid_op)
        

    return listloss,listkl


#Generate fake data for now before switching to CIFAR10
fake_data = False
#batch_size = 128
data_dir = "data/"
eval_freq = 400
num_monte_carlo = 50
architecture = "resnet" # or "vgg"
kernel_posterior_scale_mean = 0.9
kernel_posterior_scale_constraint = 0.2
kl_annealing = 50
subtract_pixel_mean = True

num_examples = 10
batch_size = args["batchsize"]
epochs = args["nbepochs"]
nb_runs = args["nbruns"]
seed0 = 23456


with tf.Session() as sess:
    if fake_data:
        (x_train, y_train), (x_test, y_test) = build_fake_data(num_examples)
        print("Using Fake DATA")
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print("Using CIFAR10 DATA")

    (images, labels, handle,
     training_iterator,
     heldout_iterator) = build_input_pipeline(x_train, x_test, y_train, y_test,
                                              batch_size, 500)

print("STARTING RUNS")

lr_adam = 0.001
adam = []
for _ in range(nb_runs):
    tf.random.set_random_seed(_*seed0)
    loss, kl = run_experiment(algo='adam', 
                         fake_data=fake_data, 
                         batch_size = batch_size, 
                         epochs=epochs,
                         learning_rate=lr_adam, 
                         verbose= True)
    adam.append(loss)
with open('losses/adam', 'wb') as fp: 
    pickle.dump(adam, fp)
print("ADAM done")

# lr_adagrad = 0.001
# adagrad = []
# for _ in range(nb_runs):
#     tf.random.set_random_seed(_*seed0)
#     loss, kl = run_experiment(algo='adagrad', 
#                          fake_data=fake_data, 
#                          batch_size = batch_size, 
#                          epochs=epochs,
#                          learning_rate=lr_adagrad, 
#                          verbose= True)
#     adagrad.append(loss)
# with open('losses/adagrad', 'wb') as fp: 
#     pickle.dump(adagrad, fp)
# print("ADAGRAD done")

# lr_adadelta = 0.0001
# adadelta = []
# for _ in range(nb_runs):
#     tf.random.set_random_seed(_*seed0)
#     loss, kl = run_experiment(algo='adadelta', 
#                          fake_data=fake_data, 
#                          batch_size = batch_size, 
#                          epochs=epochs,
#                          learning_rate=lr_adadelta, 
#                          verbose= True)
#     adadelta.append(loss)
# with open('losses/adadelta', 'wb') as fp: 
#     pickle.dump(adadelta, fp)
# print("ADADELTA done")

# lr_rmsprop = 0.0001
# rmsprop = []
# for _ in range(nb_runs):
#     tf.random.set_random_seed(_*seed0)
#     loss, kl = run_experiment(algo='rmsprop', 
#                          fake_data=fake_data, 
#                          batch_size = batch_size, 
#                          epochs=epochs,
#                          learning_rate=lr_rmsprop, 
#                          verbose= True)
#     rmsprop.append(loss)
# with open('losses/rmsprop', 'wb') as fp: 
#     pickle.dump(rmsprop, fp)
# print("MISSO done")

print("ALL LOSSES ARE SAVED")