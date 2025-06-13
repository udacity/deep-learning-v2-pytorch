# Deep Learning (PyTorch) - ND101 v7

This repository contains material related to Udacity's [Deep Learning v7 Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight initialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by real people (Udacity reviewers), but the starting code is available here, as well.

## Table Of Contents

### Tutorials

### Introduction to Neural Networks

* [Introduction to Neural Networks](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/1-Introduction%20to%20Neural%20Networks/intro-neural-networks): Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.

* [Introduction to PyTorch](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/1-Introduction%20to%20Neural%20Networks/intro-to-pytorch): Learn how to build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.

* [Project: Predicting Bike-Sharing Patterns](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/5-Projects/project-bikesharing): Apply your neural network knowledge by implementing a complete neural network in NumPy to predict bike rental patterns. This hands-on project will reinforce your understanding of neural network fundamentals.

### Convolutional Neural Networks

* [Convolutional Neural Networks](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/2-Convolutional%20Neural%20Networks/convolutional-neural-networks): Start with CNN fundamentals. First understand convolution layers through visualization, then work with multilayer perceptrons on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), and finally learn how to build CNNs for the more complex [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

* [Weight Initialization](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/2-Convolutional%20Neural%20Networks/weight-initialization): Dive deep into neural network weight initialization techniques and learn how different initialization methods affect model training and performance.

* [Batch Normalization](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/4-Generative%20Adversarial%20Networks/batch-norm): Learn this crucial deep learning technique that significantly improves network training speed and stability - a key technology for building deep CNNs.

* [Transfer Learning](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/2-Convolutional%20Neural%20Networks/transfer-learning): Learn how to leverage pre-trained models like VGGnet to solve real-world problems. This approach enables building high-performance image classifiers with limited data and computational resources.

* [Autoencoders](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/2-Convolutional%20Neural%20Networks/autoencoder): Explore a key CNN application - autoencoders. Learn how to build autoencoders using feedforward networks and CNNs for image compression and denoising.

* [Style Transfer](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/2-Convolutional%20Neural%20Networks/style-transfer): Study advanced CNN applications. Based on the paper [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), implement image style transfer by applying various CNN concepts.

* [Project: Dog Breed Classifier](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/5-Projects/project-dog-classification): Put your CNN knowledge to practice by building an end-to-end image classification pipeline. Create a CNN with PyTorch to classify dog breeds, and even detect dogs in images of humans. This project combines transfer learning, data preprocessing, and model deployment skills.

### Recurrent Neural Networks

* [Recurrent Neural Networks](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/3-Recurrent%20Neural%20Networks/recurrent-neural-networks): Begin with fundamental RNNs. First understand RNN principles through time series prediction tasks, then learn character-level RNN implementation. This will help you deeply understand how RNNs process sequential data.

* [Sentiment Analysis with NumPy](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/3-Recurrent%20Neural%20Networks/sentiment-analysis-network): (Optional) For those interested in deeper understanding, follow [Andrew Trask](http://iamtrask.github.io/)'s guide to implement sentiment analysis from scratch using NumPy. This hands-on exercise provides valuable insights into the inner workings of neural networks and text processing fundamentals.

* [Embeddings (Word2Vec)](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/3-Recurrent%20Neural%20Networks/word2vec-embeddings): Learn word embedding techniques and master how to convert text into numerical representations that neural networks can process. This is a crucial foundation for natural language processing and key to building efficient text processing models.

* [Sentiment Analysis RNN](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/3-Recurrent%20Neural%20Networks/sentiment-rnn): Combine RNN and word embedding knowledge to build a more sophisticated sentiment analysis model. This project will help you understand how to apply previously learned concepts to practical problems.

* [Attention](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/3-Recurrent%20Neural%20Networks/attention): Study the attention mechanism, a crucial concept in modern deep learning. Understand how to use attention to enhance model performance, laying the groundwork for learning advanced models like Transformers.

* [Project: TV Script Generation](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/5-Projects/project-tv-script-generation): Apply your RNN expertise to build a model that can generate TV script dialogue. Train your network on existing scripts from the TV show Seinfeld and watch it generate new, original dialogue. This project demonstrates the practical application of RNNs in creative text generation.

### Generative Adversarial Networks

* [Generative Adversarial Network on MNIST](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/4-Generative%20Adversarial%20Networks/gan-mnist): Start with basic GANs using the simple MNIST dataset to understand core GAN concepts: the adversarial training process between generator and discriminator. This introductory project will help build your foundation in GANs.

* [Deep Convolutional GAN (DCGAN)](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/4-Generative%20Adversarial%20Networks/dcgan-svhn): Learn how to apply CNNs to GANs. Using the more complex Street View House Numbers (SVHN) dataset, implement DCGAN to generate more realistic images. This project combines knowledge from both CNNs and GANs.

* [CycleGAN](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/4-Generative%20Adversarial%20Networks/cycle-gan): Study advanced GAN applications. CycleGAN can learn image transformations (like summer to winter) without paired data. This project demonstrates the powerful capabilities of GANs in practical applications.

* [Project: Face Generation](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/5-Projects/project-face-generation): Put your GAN knowledge into practice by building a DCGAN to generate realistic human faces. Using the CelebA dataset, you'll implement and train a generative model that can create entirely new, realistic faces. This project combines your understanding of CNNs and GANs in a creative application.

### Deploying a Model (with AWS SageMaker)

* [All exercise and project notebooks](https://github.com/udacity/sagemaker-deployment) for the lessons on model deployment can be found in the linked, Github repo. Learn to deploy pre-trained models using AWS SageMaker.



### Elective Material

* [Intro to TensorFlow](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/6-Elective%20Material/tensorflow/intro-to-tensorflow): Starting building neural networks with TensorFlow.
* [Keras](https://github.com/Hebe-330/deep-learning-v2-pytorch/tree/master/6-Elective%20Material/keras): Learn to build neural networks and convolutional neural networks with Keras.

---

# Dependencies

## Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
- **Mac:** https://conda.io/projects/conda/en/latest/user-guide/install/macos.html
- **Windows:** https://conda.io/projects/conda/en/latest/user-guide/install/windows.html

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

If you'd like to learn more about version control and using `git` from the command line, take a look at our [free course: Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123).

**Now, we're ready to create our local environment!**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/Hebe-330/deep-learning-v2-pytorch.git
cd deep-learning-v2-pytorch
```

2. Create (and activate) a new environment, named `deep-learning` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n deep-learning python=3.6
	source activate deep-learning
	```
	- __Windows__: 
	```
	conda create --name deep-learning python=3.6
	activate deep-learning
	```
	
	At this point your command line should look something like: `(deep-learning) <User>:deep-learning-v2-pytorch <user>$`. The `(deep-learning)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

7. That's it!

Now most of the `deep-learning` libraries are available to you. Very occasionally, you will see a repository with an addition requirements file, which exists should you want to use TensorFlow and Keras, for example. In this case, you're encouraged to install another library to your existing environment, or create a new environment for a specific project. 

Now, assuming your `deep-learning` environment is still activated, you can navigate to the main repo and start looking at the notebooks:

```
cd
cd deep-learning-v2-pytorch
jupyter notebook
```

To exit the environment when you have completed your work session, simply close the terminal window.
