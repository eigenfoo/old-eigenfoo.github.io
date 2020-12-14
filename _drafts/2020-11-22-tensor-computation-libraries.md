---
title: Anatomy of a Tensor Computation Library
excerpt: ""
tags:
  - theano
  - open source
  - machine learning
  - deep learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background12.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
toc: true
toc_sticky: true
toc_label: "Do not feed the animals"
toc_icon: "kiwi-bird"
last_modified_at: 2020-11-22
---

I was first introduced to PyTorch and TensorFlow and, having no other reference, I thought they were
the prototypical examples of tensor libraries. Then I heard about JAX (not to mention [the PyMC
developers announced that Theano would have a new JAX
backend](https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b)),
which seemed to be a GPU-friendly version of NumPy? But JAX also ships with `jit` and `vmap`, which
are [clearly first-class citizens](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).
And isn't JIT that thing that Numba does? And isn't Numba competing with Cython?

The resulting confusion prompted this blog post.

Similar to [my previous post on the anatomy of probabilistic programming
frameworks](https://eigenfoo.xyz/prob-prog-frameworks/), I'll highlight

## Dissecting Tensor Computation Libraries

First, let's take a step back and ask: what do all these tensor computation libraries do? They
provide ways of specifying and building computational graphs[^1] and

1. Running the computation itself (duh), but also running "related" computations: the most salient
   example being computing gradients.
1. Providing "best execution" for this computation: whether it's changing the execution by
   (just-in-time) compiling it, or by utilizing special hardware (GPUs/TPUs), or by vectorizing the
   computation, or in any number of different ways.
1. Optimizing the computation or allowing users to manipulate the computational graph itself: think
   symbolic simplifications (e.g.  `xy/x = y`), numerical stability (e.g. `log(1 + x)` for small
   values of `x`), or solving for `x` given `3x + 1 = 2 * (2x - 6)`.

How do these libraries differ?

1. _How_ the computational graph is built
   - Axes of variation are static vs dynamic, lazy vs eager
   - There's a complex relationship between what you want the computational graph to do, and how you
     build it. For example, TensorFlow 1.0
1. Special computations and additional features
   - E.g. deep learners want convolutions, attention mechanisms.
1. Level of abstraction and API
   - Deep learners will want Layer-oriented composability
   - This can blow up the library into a framework
   - JAX mimics NumPy as far as possible. PyTorch and TensorFlow all
   - The fewer functions you 
1. Hardware support
   - Much more prosaic, but obviously the more hardware you can support, the better.

With this in mind, we can go through the zoo of tensor computation libraries:

## A Zoo of Tensor Computation Libraries

Having outlined the basic internals of tensor computation libraries, I think it’s helpful to go
through several of the popular libraries as examples. I've tried to link to the relevant
documentation where possible.

## PyTorch and TensorFlow

## JAX
  * Trax and Flax

## Theano
  * Static graph

## Numba and Cython

Numba and Cython aren't tensor computation libraries (they don't deal with tensors), but (at least
for me) share the same headspace as all of the things that do, so it's worth disambiguating.

> PyTorch builds up a graph as you compute the forward pass, and one call to `backward()` on some
> "result" node then augments each intermediate node in the graph with the gradient of the result node
> with respect to that intermediate node. JAX on the other hand makes you express your computation as
> a Python function, and by transforming it with `grad()` gives you a gradient function that you can
> evaluate like your computation function—but instead of the output it gives you the gradient of the
> output with respect to (by default) the first parameter that your function took as input.

https://sjmielke.com/jax-purify.htm

## There is tension between higher-level programming constructs and tensor computation

It all comes down to how the library builds the computational graph, but whatever you do, there's
usually some problem with higher-level programming constructs. Whether they're obvious problems or
edge cases, you can usually expect them.

E.g. control flow

With TensorFlow, you had to compile the graph ahead of time - how can you do loops?

Conditional statements are... weird?

However, depending on how the tensor computation library works, other things may be hard too - e.g.
random number generation.

- If-Else statements
- For loops
  * https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators
  * https://theano-pymc.readthedocs.io/en/latest/library/scan.html
  * https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Control-Flow

- https://kth.instructure.com/files/1864796/download
- https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
- https://medium.com/intuitionmachine/pytorch-dynamic-computational-graphs-and-modular-deep-learning-7e7f89f18d1

---

[^1]: I think this definition is reasonable, at least for a sufficiently loose definition of
"computational graph" (i.e. if you consider computer code and call stack as a "computational
graph").
