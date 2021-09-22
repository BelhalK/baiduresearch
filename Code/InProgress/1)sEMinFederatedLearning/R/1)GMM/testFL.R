library(tensorflow)
library(reticulate)
library(tfdatasets)

py_install("tensorflow-federated")

tff <- import("tensorflow_federated")

get_local_temperature_average <- function(local_temperatures) {
  sum_and_count <- local_temperatures %>% 
    dataset_reduce(tuple(0, 0), function(x, y) tuple(x[[1]] + y, x[[2]] + 1))
  sum_and_count[[1]] / tf$cast(sum_and_count[[2]], tf$float32)
}


get_local_temperature_average <- tff$tf_computation(get_local_temperature_average, tff$SequenceType(tf$float32))



get_local_temperature_average(list(1, 2, 3))

get_global_temperature_average <- function(sensor_readings) {
  tff$federated_mean(tff$federated_map(get_local_temperature_average, sensor_readings))
}


get_global_temperature_average <- tff$federated_computation(
get_global_temperature_average, tff$FederatedType(tff$SequenceType(tf$float32), tff$CLIENTS))


library(tensorflow)
library(reticulate)
library(dplyr)
library(keras)

library(tfds)
library(tfdatasets)


tff <- import("tensorflow_federated")
collections <- import("collections", convert = FALSE)
np <- import("numpy")