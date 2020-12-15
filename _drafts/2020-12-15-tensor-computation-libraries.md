---
title: What I Wish Someone Had Told Me About Tensor Computation Libraries
excerpt: "Sometimes it feels like it's raining tensor computation libraries. In this blog post,
we’ll break down what tensor computation libraries actually are, and how they can differ. We’ll take
a detailed look at some popular libraries as examples."
tags:
  - open source
  - machine learning
  - deep learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background12.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-12-15
toc: true
toc_sticky: true
toc_label: "Do not feed the animals"
toc_icon: "kiwi-bird"
search: false
---

{% if page.noindex == true %}
  <meta name="robots" content="noindex">
{% endif %}

Sometimes it feels like it's raining tensor computation libraries.

I was first introduced to PyTorch and TensorFlow and, having no other reference, I thought they were
the prototypical examples of tensor computation libraries. Then I learnt about Theano, which was an
older and less popular project, but different than PyTorch or TensorFlow, and better in some
meaningful ways. This was followed by JAX, which seemed to be basically NumPy with more bells and
whistles (although I had trouble articulating what they were). Then came [the announcement by the
PyMC developers that Theano would have a new JAX
back-end](https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b).

Anyways, my confusion prompted a lot of research and eventually, this blog post.

Similar to [my previous post on the anatomy of probabilistic programming
frameworks](https://eigenfoo.xyz/prob-prog-frameworks/), I’ll first discuss tensor computation
libraries in general - what they are and how they can differ - and then discuss some (but certainly
not all!) tensor computation libraries in detail.

## Dissecting Tensor Computation Libraries

Firstly, a characterization: what do all these tensor computation libraries even do?

1. Provide ways of specifying and building computational graphs,
1. Running the computation itself (duh), but also running "related" computations that either (a)
   _use the computational graph_ or (b) operate _directly on the computational graph itself_
   * The most salient example of the former is computing gradients via
     [autodifferentiation](https://arxiv.org/abs/1502.05767),
   * A good example of the latter is optimizing the computation itself: think symbolic
     simplifications (e.g.  `xy/x = y`) or modifications for numerical stability (e.g. [`log(1 + x)`
     for small values of `x`](https://cs.stackexchange.com/q/68411)).
1. Providing "best execution" for this computation: whether it's changing the execution by
   (just-in-time) compiling it, or by utilizing special hardware (GPUs/TPUs), or by vectorizing the
   computation, or in any number of different ways,

---

As an aside, I realize that the name "tensor computation library" is overly broad, and that the
characterization above precludes some libraries that might also be called "tensor computation
libraries". So, for the avoidance of doubt, here is a list of libraries that this blog post is _not_
about:

- NumPy and SciPy
  * These libraries don't have a concept of a computational graph - they’re more like a toolbox of
    functions, called from Python but executed in C or Fortran.
  * However, this might be a pretty controversial distinction - as we’ll see later, JAX also doesn't
    build an explicit computational graph either, and I definitely want to include JAX as a "tensor
    computation library"... `¯\_(ツ)_/¯`
- Numba and Cython
  * These libraries provide best execution for code (and in fact some tensor computation libraries,
    such as Theano, use them to ensure "best execution"), but like NumPy and SciPy, they do not
    actually manage the computation itself.
- Keras, Trax, Flax and PyTorch-Lightning
  * These libraries are high-level wrappers around tensor computation libraries - they basically
    provide abstractions and a user-facing API to utilize tensor computation libraries in a
    friendlier way.

---

All three goals are fairly ambitious undertakings that have sophisticated solutions, and it
shouldn't be surprising to learn that design decisions in pursuit on goal can have implications for
(or even incur a trade-off with!) the other goals. Here's a (non-exhaustive) list of common
differences along all three axes:

1. Tensor computation libraries can differ in how they represent the computational graph, and how it
   is built.
   - Static or dynamic graphs: do we first define the graph completely and then inject data to run
     (a.k.a. define-and-run), or is the graph defined on-the-fly via the actual forward computation
     (a.k.a. define-by-run)?
     * TensorFlow 1.0 was (in)famous for its static graphs, which made users feel like they were
       "working with their computational graph through a keyhole", especially when [compared to
       PyTorch's dynamic graphs](https://news.ycombinator.com/item?id=13429355).
   - Lazy vs eager execution: do we evaluate variables as soon as they are defined, or only when a
     dependent variable is evaluated? Usually, tensor computation libraries either choose to support
     dynamic graphs with eager execution, or static graphs with lazy execution - for example,
     [TensorFlow 2.0 supports both modes](https://www.tensorflow.org/guide/eager).
   - Interestingly, some tensor computation libraries (e.g. [Thinc](https://thinc.ai/)) don't even
     construct an explicit computational graph: they represent it as [chained higher-order
     functions](https://thinc.ai/docs/concept).

1. Tensor computation libraries can also differ in what they want to use the computational graph
   _for_ - for example, are we aiming to do things that basically amount to running the
   computational graph in a "different mode", or are we aiming to modify the computational graph
   itself?
   - Almost all tensor computation libraries support autodifferentiation in some capacity (either
     forward-mode, backward-mode, or both).
   - Obviously, how you represent the computational graph and what you want to use it for are very
     related questions - for example, if you want your computational graph to represent "arbitrary
     computation", it's hard to do control flow such as if-else statements or for-loops - this leads
     to common gotchas with [using Python for-loops in
     JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Control-Flow)
     or [needing to use `torch.nn.ModuleList` in for-loops with
     PyTorch](https://discuss.pytorch.org/t/can-you-have-for-loops-in-the-forward-prop/68295)
   - Some tensor computation libraries (e.g. [Theano](https://github.com/Theano/Theano) and it's
     fork, [Theano-PyMC](https://theano-pymc.readthedocs.io/en/latest/index.html)) aim to [optimize
     the computational graph
     itself](https://theano-pymc.readthedocs.io/en/latest/extending/optimization.html)

1. Finally, tensor computation libraries can also differ in how they execute code.
   - All tensor computation libraries run on CPU, but the strength of GPU/TPU support is a major
     differentiator among tensor computation libraries.
   - Another differentiator is how tensor computation libraries compile code to be executed on
     hardware. For example, do they use just-in-time (a.k.a. JIT) compilation or not? Do they use
     "regular" C or CUDA compilers, or [the XLA compiler for machine-learning specific
     code](https://tensorflow.google.cn/xla)?

## A (Small) Zoo of Tensor Computation Libraries

Having outlined the basic similarities and differences of tensor computation libraries, I think
it’ll helpful to go through several of the popular libraries as examples. I’ve tried to link to the
relevant documentation where possible. [^1]

### [PyTorch](https://pytorch.org/)

1. How is the computational graph is represented and built?
   - PyTorch dynamically builds (and eagerly evaluates) an explicit computational graph. For more
     detail on how this is done, check out [the PyTorch docs on autograd
     mechanics](https://pytorch.org/docs/stable/notes/autograd.html),
   - For more on how PyTorch computational graphs, see [`jdhao`'s introductory blog post on
     computational graphs in
     PyTorch](https://jdhao.github.io/2017/11/12/pytorch-computation-graph/),
1. What is the computational graph used for?
   - To quote the [PyTorch docs](https://pytorch.org/docs/stable/index.html), "PyTorch is an
     optimized tensor library for deep learning using GPUs and CPUs" - as such, the main focus is on
     [autodifferentiation](https://pytorch.org/docs/stable/notes/autograd.html).
1. How does the library ensure "best execution" for computation?
   - PyTorch has [strong GPU support](https://pytorch.org/docs/stable/notes/cuda.html) via CUDA.
   - PyTorch also has support for TPU through projects like
     [PyTorch/XLA](https://github.com/pytorch/xla) and
     [PyTorch-Lightning](https://www.pytorchlightning.ai/)

### [JAX](https://jax.readthedocs.io/en/latest/)

1. How is the computational graph is represented and built?
   - Instead of building an explicit computational graph to compute gradients, JAX simply supplies a
     `grad()` that returns the gradient function of any supplied function. As such, there is
     technically no concept of a computational graph - only pure (i.e. stateless and
     side-effect-free) functions and their gradients.
   - [Sabrina Mielke summarizes the situation very well](https://sjmielke.com/jax-purify.htm):

     > PyTorch builds up a graph as you compute the forward pass, and one call to `backward()` on
     > some "result" node then augments each intermediate node in the graph with the gradient of the
     > result node with respect to that intermediate node. JAX on the other hand makes you express
     > your computation as a Python function, and by transforming it with `grad()` gives you a
     > gradient function that you can evaluate like your computation function — but instead of the
     > output it gives you the gradient of the output with respect to (by default) the first
     > parameter that your function took as input.

1. What is the computational graph used for?
   - According to the [JAX quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html),
     JAX bills itself as "NumPy on the CPU, GPU, and TPU, with great automatic differentiation for
     high-performance machine learning research". Hence, it's focus is heavily on
     autodifferentiation.
1. How does the library ensure "best execution" for computation?
   - This is best explained by quoting the [JAX quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html):

     > JAX uses XLA to compile and run your NumPy code on [...] GPUs and TPUs. Compilation happens
     > under the hood by default, with library calls getting just-in-time compiled and executed. But
     > JAX even lets you just-in-time compile your own Python functions into XLA-optimized kernels
     > [...] Compilation and automatic differentiation can be composed arbitrarily [...]

   - For more detail on JAX’s four-function API (`grad`, `jit`, `vmap` and `pmap`), see
     [Alex Minaar's overview of how JAX works](http://alexminnaar.com/2020/08/15/jax-overview.html).

### [Theano](https://theano-pymc.readthedocs.io/en/latest/)

> **Note:** the [original Theano](https://github.com/Theano/Theano) (maintained by [MILA) has been
> discontinued, and the PyMC developers have forked the project -
> [Theano-PyMC](https://github.com/pymc-devs/Theano-PyMC) (soon to be renamed Aesara). I'll discuss
> both the original and forked projects below.

1. How is the computational graph is represented and built?
   - Theano statically builds (and lazily evaluates) an explicit computational graph.
1. What is the computational graph used for?
   - Theano is unique among tensor computation libraries in that places more emphasis on reasoning
     about the computational graph itself. In other words, while Theano has [strong support for
     autodifferentiation](https://theano-pymc.readthedocs.io/en/latest/library/gradient.html),
     running the computation and computing gradients isn't the be-all and end-all: Theano has an
     entire module for [optimizing the computational graph
     itself](https://theano-pymc.readthedocs.io/en/latest/optimizations.html), and makes it fairly
     straightforward to compile the Theano graph to different computational back-ends (by default,
     Theano compiles to C or CUDA, but it’s straightforward to compile to JAX).
   - Theano is often remembered as a library for deep learning research, but it’s so much more than
     that!
1. How does the library ensure "best execution" for computation?
   - The original Theano used the GCC C compiler for CPU computation, and the NVCC CUDA compiler for
     GPU computation.
   - The Theano-PyMC fork project [will use JAX as a
     backend](https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b),
     which can utilize CPUs, GPUs and TPUs as available.

---

[^1]: Some readers will notice the conspicuous lack of TensorFlow from this list - its exclusion isn't out of malice, merely a lack of time and effort to do the necessary research to do it justice. Sorry.
