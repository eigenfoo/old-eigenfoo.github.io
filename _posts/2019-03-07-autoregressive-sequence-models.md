---
title: Deep Autoregressive Generative Sequence Models
excerpt:
tags:
  - deep learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background14.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at:
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
      these models are to compare them to recurrent neural networks (RNNs),
      which are a lot more well-known.

    <figure>
        <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png"><img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png"></a>
        <figcaption>Obligatory RNN diagram. Source: <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Chris Olah</a>.</figcaption>
    </figure>

    * Like a recurrent model, an autoregressive model's output $$h_t$$ at time
      $$t$$ depends on not just $$x_t$$, but also $$x$$'s from previous time
      steps. However, unlike a recurrent model, the previous $$x$$'s are not
      provided via some hidden state: they are given as just another input to
      the model.
    * To train this model, one unfolds the recurrent network (similar to
      what is depicted above), and performs gradient descent on the outputs.
      This algorithm is given the somewhat misleading name ["backpropagation
      through
      time"](https://en.wikipedia.org/wiki/Backpropagation_through_time).[^1]
    * Put simply, **an autoregressive model is merely a recurrent model in which
      there is no hidden state, and therefore one does not backpropagate through
      time.**

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
good degree of success in the real world:

 - [Wavenet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
 - [PixelCNN](https://arxiv.org/abs/1601.06759) and [PixelCNN++](https://arxiv.org/abs/1701.05517)
 - [Transformer](https://arxiv.org/abs/1706.03762)


## Some Observations

 - These models model the _likelihood_ of data. They can do by modelling the
   likelihood function directly (a simple task when the likelihood is discrete),
   or by modelling the parameters of some pdf.

 - These models are supervised learning. With the success of GANs and VAEs, it
   is easy to assume generative models must be unsupervised learning. This is
   not true! Modelling the likelihood is what allows this to be supervised.

 - None of these models worry about "stopping". Audio and images have a fixed
   number of time steps: generate $$N$$ audio samples, or $$N^2$$ pixel values.
   Text is a bit different: thankfully it is discrete, so we can have one more
   category to indicate "stop". None of these models have both a variable number
   of outputs _and_ continuous inference variables.

https://deepgenerativemodels.github.io/notes/
https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
https://bair.berkeley.edu/blog/2018/08/06/recurrent/

---

[^1]: There is some nuance here about truncation, but whatever.
