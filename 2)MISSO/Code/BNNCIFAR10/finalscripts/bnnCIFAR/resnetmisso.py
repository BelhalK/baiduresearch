from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import flags
import matplotlib
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from models.bayesian_resnet import bayesian_resnet
from models.bayesian_vgg import bayesian_vgg


def run_experiment_misso(algo,fake_data, batch_size, epochs, verbose):
    with tf.Session() as sess:
        
        model_fn = bayesian_resnet
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
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        
        train_op = opt.minimize(loss)
        update_step_op = tf.compat.v1.assign(t, t + 1)

        with tf.compat.v1.name_scope("valid"):
          valid_accuracy, valid_accuracy_update_op = tf.compat.v1.metrics.accuracy(
              labels=labels, predictions=predictions)

        init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                           tf.compat.v1.local_variables_initializer())

        stream_vars_valid = [
            v for v in tf.compat.v1.local_variables() if "valid/" in v.name
        ]
        reset_valid_op = tf.compat.v1.variables_initializer(stream_vars_valid)
    
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        indivgrads = []
        indivvar = []
        # Run the training loop
        train_handle = sess.run(training_iterator.string_handle())
        heldout_handle = sess.run(heldout_iterator.string_handle())
        training_steps = int(
          round(epochs * (len(x_train) / batch_size)))
        listloss = []
        listaccuracy = []
        print(training_steps)
        for indiv in range(0,total,batch_size):
            print(indiv)
            grads = tf.gradients(loss, tf.trainable_variables())
            var_updates = []
            var_list = tf.trainable_variables()
            for grad, var in zip(grads, var_list):
                var_updates.append(var.assign_sub(0.001 * grad))
            train_op = tf.group(*var_updates)
            indivgrads.append(grads)
            indivvar.append(var_list)
#        for step in range(training_steps):
        for epoch in range(iteration):
            for step in range(0,int(total/batch_size)):
                grads = tf.gradients(loss, tf.trainable_variables())
                indivgrads[step] = grads
                var_updates = []
                var_list = tf.trainable_variables()
                print('ok')
                for gradstemp, varlist in zip(indivgrads, indivvar):
                    for grad, var in zip(gradstemp, varlist):
                        var_updates.append(var.assign_sub(0.001 * grad)) #\theta^{\tau_i^k} - \grad f_{\tau_i^k}
                _ = sess.run([train_op,
                      train_accuracy_update_op,
                      update_step_op],
                     feed_dict={handle: train_handle})
                # Print loss values
                loss_value, accuracy_value, kl_value = sess.run(
                  [loss, train_accuracy, kl], feed_dict={handle: train_handle})
                print(
                  "Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f} KL: {:.3f}".format(
                      step, loss_value, accuracy_value, kl_value))
                listloss.append(loss_value)
                listaccuracy.append(accuracy_value)
                
                if (step + 1) % eval_freq == 0:
              # Compute log prob of heldout set by averaging draws from the model:
              # p(heldout | train) = int_model p(heldout|model) p(model|train)
              #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
              # where model_i is a draw from the posterior
              # p(model|train).
                  probs = np.asarray([sess.run((labels_distribution.probs),
                                           feed_dict={handle: heldout_handle})
                                  for _ in range(num_monte_carlo)])
                  mean_probs = np.mean(probs, axis=0)

                  _, label_vals = sess.run(
                      (images, labels), feed_dict={handle: heldout_handle})
                  heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                     label_vals.flatten()]))
                  print(" ... Held-out nats: {:.3f}".format(heldout_lp))

          # Calculate validation accuracy
          #for _ in range(20):
           # sess.run(
            #    valid_accuracy_update_op, feed_dict={handle: heldout_handle})
          #valid_value = sess.run(
           #   valid_accuracy, feed_dict={handle: heldout_handle})

    #      print(" ... Validation Accuracy: {:.3f}".format(valid_value))
        sess.run(reset_valid_op)
    return listloss
    