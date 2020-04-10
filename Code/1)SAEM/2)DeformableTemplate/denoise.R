load("usps1.RData")
library(MASS)
library(ggplot2)
library(reshape2)
require(ggplot2)
require(gridExtra)
require(reshape2)
library(rlist)
require(RnavGraphImageData)
theme_set(theme_bw())
options(digits = 2)

library(imagerExtra)
library("OpenImageR")
library('magick')

#imagerExtra#
gd <- grayscale(dogs)
plot(gd, main = "dogs")
my.image = tail(newsamples.inc,1)[[1]] + meantemp
image(t(my.image)[,nrow(my.image):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))


#magick#
image <- magick::image_read('test.png')
image = image_equalize(image)
plot(image_contrast(image, sharpen = 10))