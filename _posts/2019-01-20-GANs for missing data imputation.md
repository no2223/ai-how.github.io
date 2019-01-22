---
layout: post
title: Learning missing value distribution and its imputation using GANs
published: false
comments: true
image: "/img/MAML.jpg"
share-img: "/img/MAML.jpg"
thumbnail: "/img/MAML.jpg"
---

# GANs quick recap

GANs as a generative modeling approach has demonstrated the ability to learn an underlying distribution of the partially observed data. This learnt distribution is then applied to generate sample of similar kind thereby benefitting in increasing the data size and its quality. GANs have found useful in several applications (not limited to):
  * Image enhancement such as super resolution
  * Style transfer
  * Healthcare such as drug discovery

In this blog lets walkthrough how GANs are used to impute missing observations also known as incomplete data. More specifically I am referring to recent article published at ICLR 2019 [1](https://openreview.net/pdf?id=S1lDV3RcKm)

It utilizes following three pairs of Generators and discriminators:
   * learning the distirbution of missing values
   * learning the distribution of incomplete target data
   * Imputer which imputes the missing observation in observed dataset

# How it works ?

Given a set of incomplete observations, a mask is defined as below (zero if observation is missing and 1 otherwise).
                                                    m Є {0,1}

An incomplete dataset can now be represented as:
                    
                                           D = {(xi, mi)} where i Є 1,...,N

<p align="center"> <img src="https://ai-how.github.io/img/20190120_181543.png" width="400" height="300" /> </p>

Lets talk about how GANs learns the distribution of location of missing values in an incomplete dataset. Given an input sampled from a distribution known apriori, generater Gmask generates samples which are then fed to discriminator along with masked data derived from observed data. Masked data contains 1s in place where data is observed and 0 otherwise. In this process generator learns to map the apriori distribution to a masked data distribution and adjust its weights so as to be able to locate position of missing values in an observed data.

Loss for this GAN (am referring here as GANmask) is defined as below:

<p align="center"> <img src="https://ai-how.github.io/img/Mask_Loss.png" width="450" height="25" /> </p>

Another set of GAN (referred as Gdata & Ddata) learns to generate complete observation using incomplete dataset. As usual with GAN training, generator here learns to generate samples (closer to the incomplete data) through an adversarial process where discriminator attempts hard to discriminate between generated samples and observed data. The input to the discriminator is slightly modified using masked operator (defined below):

<p align="center"> <img src="https://ai-how.github.io/img/Disc_Inp.png" width="200" height="25" /> </p>

Here ̃m represents the compliment of m and is used to fill location of missing values with a constant. The training objective of this GAN now becomes:

<p align="center"> <img src="https://ai-how.github.io/img/GANdata_Loss.png" width="550" height="25" /> </p>

As seen above the combined output of Gmask and Gdata becomes fake sample for discriminator to discriminate against the real data slightly modified using the masking operator (shown above).

Thus the final training objective of GANmask and GANdata now becomes as below:

<p align="center"> <img src="https://ai-how.github.io/img/Combined_Objective.png" width="300" height="50" /> </p>

# Imputing missing values in observed data

With the above configuration GANs now would be able to learn the distribution of missing values and generate complete data. Something that interests a lot is to be able to impute missing values in place on a given incomplete data. For this another GAN known as imputer is trained (depicted below):

<p align="center"> <img src="https://ai-how.github.io/img/20190120_181402.png" width="400" height="300" /> </p>

Generator (Gimpute here) takes real data (incomplete) along with some noise and learns to generate samples with following characteristics:

   * observed values are kept intact
   * missing locations are imputed

Whereas the discriminator learns to discriminate between generated samples (that comes from Gdata) and Gimpute. For imputer GAN loss is defined as below:

<p align="center"> <img src="https://ai-how.github.io/img/Loss_Imputer.png" width="500" height="25" /> </p>

The joint learning for generating process and imputer is defined as according to the following objectives:

<p align="center"> <img src="https://ai-how.github.io/img/Joint_Learning.png" width="500" height="90" /> </p>

The whole architectural approach to use GANs for missing value imputation looks quite convincing. Lets have a look at how this approach compares against some of the benchmarks. Below depcicts the performance comparison of how well MisGAN (as referred here in this paper [1](https://openreview.net/pdf?id=S1lDV3RcKm)) imputes missing value and is able to generate clear digits where other methods looks to struggle.

For more indepth details I would recommend to read this interesting paper which got accepted at ICLR-2019 [1](https://openreview.net/pdf?id=S1lDV3RcKm)

to generate synthetic sample coming from the To mimic the way human perceive, process and associate the previously seen information to better identify the current experience requires ability to extend the information learnt from previous tasks. Human brain is good at propagating the information learnt from one task to adapt to another. As a result, it quickly grasp and understand the concepts with minimal number of examples.

Current deep learning approaches are capable of learning any task provided labeled data is available in abundance and performs poorly where limited training dataset is available. It is therefore of great significance to develop learning algorithms that leverages information gained from a related task and learns efficiently and rapidly when presented with a limited number of training examples (also known as few shot problem).

To address few shot learning meta learning methods have been proposed, broadly categorized as (very well described in [1](https://arxiv.org/pdf/1807.05960.pdf)):

* Metric based methods (focuses on learning similarity metrics between members of the same class)
* Memory based methods (exploits memory baseed architecture to store key training examples)
* Optimization based methods (search for an optimal region in parameter space favorable to fast adaptation of new task)

This blogpost aims to explain the implementation of optimization based method (Specifically known as MAML proposed in [2](https://arxiv.org/pdf/1703.03400.pdf)) applied in the context of few shot learning. MAML aims to find the optimal point in the parameter space which when presented with multiple tasks (limited training examples for each of the task) can easily be adapted to each of these task with few gradient updates and limited training example. Figure below depicts the point in a hyperparameter space (θ of a deep neural network) where the network can easily be trained to each of the three given tasks corresponding to three different directions (θ1, θ2, θ3).

<p align="center"> <img src="https://ai-how.github.io/img/MAML.jpg" width="300" height="200" /> </p>

Thus in a meta-learning phase the model aims to identify the parameter Ɵ such that when presented with a new task (or a set of tasks) the parameters can easily be tuned in a few gradient steps.

In training phase the parameter set θ is obtained by evaluating the validation loss over a set of task given as:

<p align="center"> <img src="https://ai-how.github.io/img/theta.jpg" width="300" height="50" /> </p>

whereas task specific θ's (such as θ1, θ2, θ3) are obtained as follows (by performing few gradient steps independently over each task using θ as initial starting point):

<p align="center"> <img src="https://ai-how.github.io/img/thetadash.jpg" width="200" height="50" /> </p>

This approach has shown to work well for few shot classification problem such as mini-imagenet. The latest research in this direction is discussed in [3](https://arxiv.org/pdf/1807.05960.pdf) and improves by a large margin over MAML.

# When to use ?

In various industrial application there exist dataset which is not in abundance to fullfill the deep neural network desire to train effectively. Meta learning in such domains become the natural choice to train models which could learn from limited dataset available for different tasks quickly and efficiently. Say a credit card company launhces a new credit card across selective few markets and the datapoints depicting defaulter behavior is emerging (and thus very few), in such scenarios building predictive model using limited datapoints would not work in a traditional way. With meta learning approach each market could be treated as a task and thus the model parameter for each task can be fine-tuned to model defaulter behavior efficiently.

Research along this direction forces me to think if existing classification models (for large labeled training dataset) build this way would be more efficient or not ?
