### Date Created: October 23rd 2018

# Denoising Autoencoder
Autoencoder is a simple script which defines a convolutional autoencoder using a set of images from NASA's earth
observatory. An autoencoder has two main components, an encoder which compresses input data and a decoder that
reconstructs data from its compressed representation. Autoencoders are useful because they "learn" how to reduce the
dimensionality of any input. This allows one to use the compressed representation faster than they could
using the original input. Autoencoders are useful for image denoising, filling in missing data and generative models.

## Description
This repo contains the code can be used to get you started with Autoencoders.  More specifically, it walks one through
an alternative "denoising" architecture which starts with a depth of 32 and uses convolutional layers instead of transposed convolutional layers. The example includes random noise used for each image. We use the noise as
an input to the denoiser and the loss function is based on the comparison of the reconstructed images with our original
images. The final step of the code compares the image reconstruction with the originals.

## Installation
* clone the hit hub project
* cd into the project folder

## Creators
* Anton Gordon but based on documents from Udacity's Deep Learning Nanodegree.
