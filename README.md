# Deep Learning (PyTorch)

This repository contains material related to Udacity's [Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight intialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by real people (Udacity reviewers), but the starting code is available here, as well.

## Table Of Contents

### Tutorials

* [Introduction to Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-neural-networks): Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.
* [Sentiment Analysis with NumPy](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.
* **This repo is in development (as of 9/10) and there are many more tutorials to come!**


### Projects

* [Predicting Bike-Sharing Patterns](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing): Implement a neural network in NumPy to predict bike rentals.
* [Dog Breed Classifier](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification): Build a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.
* More to come

### Elective Material

* [Intro to TensorFlow](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/tensorflow/intro-to-tensorflow): Starting building neural networks with Tensorflow.
* More to come

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

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

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
git clone https://github.com/udacity/deep-learning-v2-pytorch.git
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

Noe, assuming your `deep-learning` environment is still activated, you can navigate to the main repo and start looking at the notebooks:

```
cd
cd deep-learning-v2-pytorch
jupyter notebook
```

To exit the environment when you have completed your work session, simply close the terminal window.
