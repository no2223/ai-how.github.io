---
layout: post
title: From Supervised learning to semi-supervised and unsupervised approach 
published: false
comments: true
---
## Brief to Generative and Discriminative models
Learning approaches can be broadly divided into two categories; generative and discriminative. Generative models learn the underlying distribution of data representing various categories for example, model may learn distribution of pixels in an image when a cat is present differently in comparison to presence of a dog. Thus, training generative model involves identifying following:
<p align="center"> <img src="https://github.com/ai-how/ai-how.github.io/blob/master/img/gen.png" /> </p>

Ideally once the model learns the real underlying distribution it can be used to generate unseen samples which could further improve generalization capability of supervised techniques (discussed later). Discriminative models on the other hand models the conditional probability via relying heavily on the observed dataset for example, given the pixels arrangement in an image it learns to label an image as a cat or dog (considering two label calssification problems)
<p align="center"> <img src="https://github.com/ai-how/ai-how.github.io/blob/master/img/cond.png" /> </p>

Generative models aids the unsupervised learning by improving the learned representations of observed samples corresponding to different classes.
## Improving learning using unsupervised and supervised techniques
Before I explain how unsupervised learning can augment the supervised technique (or a classifier), I am tempted to quote following which fules the need to reap the benefits unsupervised learning can deliver
>_If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake. We know how to make the icing and the cherry, but we don’t know how to make the cake_
### How to improve generalization using unsupervised technique?
Unsupervised learning involves learning the representaions from observed samples with no information about their category or class. It tries to cluster the samples through its n-dimensional representations for example using auto-encoders, GANs. Assuming the task is to learn a classifier to distinguish between types of flowers, it proceeds in two steps:
* learn the n-dimensional representaion from given samples in an unsupervised manner
  * Using learned representaions train a classifier to classify images of different types

To learn the representaions in an unsupervised fashion, GANs can be first be trained to simulate images of flower by learning their underlying distribution (depcited below).

<p align="center"> <img src="https://indico.lal.in2p3.fr/event/3487/?view=standard_inline_minutes" /> </p>

$$a^2 + b^2 = c^2$$

{% raw %}
  $$a^2 + b^2 = c^2$$ --> note that all equations between these tags will not need escaping! 
 {% endraw %}

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$
