---
title: What I Wish Someone Had Told Me About Tensor Computation Libraries
excerpt: ""
tags:
  - theano
  - open source
  - machine learning
  - deep learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background12.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-11-22
---

## What even is all this stuff?

I had used PyTorch and TensorFlow, which I thought were the prototypical examples of tensor
libraries - after all, 

JAX, but JAX also ships with `jit` and `vmap`, which are [clearly first-class
citizens](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html). And isn't JIT that thing
that Numba does?

What do they all do? They provide ways of specifying and building computational graphs[^1] and

1. Running the computation itself (duh), but also running other computations (the most important
   example is taking gradients)
2. Manipulating or optimizing the computation itself (think symbolic
3. Providing the "best execution" for this computation (whether it's changing the runtime by JITting
   it, or by sending it to a GPU/TPU)

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

- PyTorch and TensorFlow
- JAX
  * Trax and Flax
- Theano
  * Static graph
- Numba and Cython aren't tensor computation libraries (they don't deal with tensors), but (at least
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

## Don't fret about understanding how autodifferentiation works

Certainly it's good to know what autodifferentiation is and how it's different from numerical or
symbolic differentiation, but I probably put too much effort into understanding the 

https://dl.acm.org/doi/abs/10.5555/3122009.3242010

Just read about forward-mode autodifferentiation: what it is, why it's not great, why backward-mode
autodifferentiation 

- Forward mode auto differentiation: read about what it is, why it's not that popular (especially
  with the kinds of computation that we want to do these days)
- Backward mode auto differentiation
- Then think of differentiation as a higher order operation - we don't care about how it's done, we
  just need `df = f.grad()`.

---

[^1]: I think this definition is reasonable, at least for a sufficiently loose definition of
"computational graph" (i.e. if you consider computer code and call stack as a "computational
graph").

