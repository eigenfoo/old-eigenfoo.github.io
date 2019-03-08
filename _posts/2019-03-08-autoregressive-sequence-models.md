---
title: Deep Autoregressive Generative Sequence Models
excerpt:
tags:
  - deep learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background14.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at: 2019-03-08
---

A current project of mine involves working with fairly niche and interesting
neural networks that aren't seen very often on a first pass through deep
learning. I thought I'd write up my research and post it.

## Deep Autoregressive Generative Sequence Models

To be concise as possible, this blog post is about _deep autoregressive
generative sequence models_. That's quite a mouthful of jargon, so let's unpack
that.

- Deep
    * Well, I'm using PyTorch... so I guess it's "deep" :wink:
- Autoregressive
    * [Stanford has a good introduction](https://deepgenerativemodels.github.io/notes/autoregressive/)
      to autoregressive generative models, but I think a good way to explain
      these models is to compare them to recurrent neural networks (RNNs), which
      are a lot more well-known.

    <figure>
        <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png"><img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png"></a>
        <figcaption>Obligatory RNN diagram. Source: <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Chris Olah</a>.</figcaption>
    </figure>

    * Like a recurrent model, an autoregressive model's output $$h_t$$ at time
      $$t$$ depends on not just $$x_t$$, but also $$x$$'s from previous time
      steps. However, unlike a recurrent model, the previous $$x$$'s are not
      provided via some hidden state: they are given as just another input to
      the model.
    * The following animation of Google DeepMind's WaveNet illustrates this
      well: the $$t$$th output is generated in a _feed-forward_ fashion from
      several input $$x$$ values.[^1]

    <figure>
        <a href="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif"><img src="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif"></a>
        <figcaption>WaveNet animation. Source: <a href="https://deepmind.com/blog/wavenet-generative-model-raw-audio/">Google DeepMind</a>.</figcaption>
    </figure>

    * Put simply, **an autoregressive model is merely a sequential model in
      which one predicts future values from past values.**

- Generative
    * Informally, a generative model is one that can generate new data after
      being trained on the dataset.
    * More formally, a generative model models the joint distribution
      $$P(X, Y)$$ of the observation $$X$$ and the target variable $$Y$$.
      Contrast this to a discriminative model that models the conditional
      distribution $$P(X|Y)$$.
    * Generative adversarial networks (GANs), variational autoencoders (VAEs)
      and normalizing flow models are all examples of generative models.

- Sequence model
    * Although sequence models are, well, sequential (duh), there has been
      success at applying them to non-sequential data. For example, PixelRNN and
      PixelCNN can generate entire images, pixel by pixel! The pixels form a
      sequence.
    * E.g. any NLP model.

A good distinction is that "generative" and "sequential" describe _what_ these
models do, or what kind of data they deal with. "Autoregressive" describes _how_
these models do what they do: i.e. they describe properties of the network or
its architecture.

Despite the technicality involved in describing these models, they have seen a
good degree of success in the real world. Each model merits discussion, but
unfortunately there isn't enough space to devote to a detailed discussion about
them.

 - [PixelCNN](https://arxiv.org/abs/1601.06759) by Google DeepMind was probably
   the first such model, and the progenitor of most of the other models below.
   The authors demonstrated that images were sequential too. This was a
   reduction from PixelRNN.
 - [PixelCNN++](https://arxiv.org/abs/1701.05517) by OpenAI was an improvement
   on PixelCNN.
 - [Wavenet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) by
   Google DeepMind was
 - [Transformer, a.k.a. _the "attention is all you need"
   model_](https://arxiv.org/abs/1706.03762)

These models also have uses in specific applications, such as [neural machine
translation in linear time](https://arxiv.org/abs/1610.10099) and
[video](https://arxiv.org/abs/1610.00527).

## Some Observations

### Modelling the likelihood

 - These models model the _likelihood_ of data. They can do by modelling the
   likelihood function directly (a simple task when the likelihood is discrete),
   or by modelling the parameters of some pdf.

### Supervised learning!

 - These models are supervised learning. With the success of GANs and VAEs, it
   is easy to assume generative models must be unsupervised learning. This is
   not true! Modelling the likelihood is what allows this to be supervised.

Way more stable than GANs, and can use all the good stuff from ML101 - cross
validation, loss metrics, etc.

### Autoregressive models work for both continuous and discrete data

Unlike GANs, which have a hard time learning discrete data.

### Stopping problem

 - None of these models worry about "stopping". Audio and images have a fixed
   number of time steps: generate $$N$$ audio samples, or $$N^2$$ pixel values.
   Text is a bit different: thankfully it is discrete, so we can have one more
   category to indicate "stop". None of these models have both a variable number
   of outputs _and_ continuous inference variables.

### Modelling multiple timescales

E.g. sound. Important co

<figure>
    <a href="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig1-Anim-160908-r01.gif"><img src="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig1-Anim-160908-r01.gif"></a>
    <figcaption>Audio exhibits patterns at multiple timescales. Source: <a href="https://deepmind.com/blog/wavenet-generative-model-raw-audio/">Google DeepMind</a>.</figcaption>
</figure>



Context stacks in WaveNet, or multi-scale PixelRNN.

### It's amazing that this all works!

> **Recurrent models trained in practice are effectively feed-forward.** This
> could happen either because truncated backpropagation time cannot learn
> patterns significantly longer than $$k$$ steps, or, more provocatively,
> because models _trainable by gradient descent_ cannot have long-term memory.
> - [John Miller](http://www.offconvex.org/2018/07/27/approximating-recurrent/)

---

[^1]: There's actually a lot more nuance than meets the eye in this animation, but all I'm trying to illustrate is the feed-forward nature of autoregressive models.
