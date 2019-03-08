---
title: "Journal Club: Deep Autoregressive Sequence Models"
excerpt:
tags:
  - deep learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background1.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at:
---

A current project of mine involves working with fairly niche and interesting
neural networks that aren't seen very often on a first pass through deep
learning. I thought I'd write up my research and post it. To be concise as
possible, this blog post is about _deep autoregressive generative sequence
models_. That's quite a mouthful of jargon, so let's unpack that.

- Deep
    * I'm using PyTorch. Therefore, "deep". :wink:
- Autoregressive
    * Show obligatory "recurrent neural network" picture here.
    * https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png
    * An autoregressive model is a recurrent model in which one does not
      backpropagate through time.
- Generative
    * For a discriminative model such as logistic regression, the fundamental
      inference task is to predict a label for any given datapoint. Generative
      models, on the other hand, learn a joint distribution over the entire
      data.
    * E.g. GANs, VAEs, normalizing flow models.
- Sequence model
    * The main difficulty is knowing when to stop! This is not a problem if the
      values you are inferring are discrete (e.g. NLP solves this problem by
      having a stop token). If values are continuous however, this is trickier.
    * Note: although sequence models are, well, sequential (duh), there has been
      success at applying them to non-sequential data. E.g. PixelRNN and
      PixelCNN can generate entire images, pixel by pixel! The pixels form a
      sequence.
    * E.g. any NLP model.

The first two (generative and sequence) are to do with _what_ these models do, or
what data they deal with. The last one (autoregressive) is to do with _how_ they
do what they do: i.e. they describe properties of the network or its architecture.

Interesting things to note:

 - These models model the _likelihood_ of data. They can do by modelling the
   likelihood function directly (a simple task when the likelihood is discrete),
   or by modelling the parameters of some pdf.
 - These models are supervised learning. With the success of GANs and VAEs, it
   is easy to assume generative models must be unsupervised learning. This is
   not true! Modelling the likelihood is what allows this to be supervised.
 - None of these models worry about "stopping". Audio and images have a fixed
   number of time steps: generate N audio samples, or N^2 pixel values. Text is
   a bit different: thankfully it is discrete, so we can have one more category
   to indicate "stop". None of these models have both a variable number of
   outputs _and_ continuous inference variables.


More in depth:

 - Wavenet
 - PixelCNN and PixelCNN++
 - Transformer

https://deepgenerativemodels.github.io/notes/
https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
https://bair.berkeley.edu/blog/2018/08/06/recurrent/
