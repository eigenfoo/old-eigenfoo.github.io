---
title: Programmatically Manipulating Python ASTs
excerpt:
tags:
  - pymc
  - python
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background1.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-03-26
search: false
---

{% if page.noindex == true %}
  <meta name="robots" content="noindex">
{% endif %}

This was originally an exercise in PyMC4's model specification API, but there are plenty
of lessons to be learnt here about manipulating Python ASTs (abstract syntax trees) more
generally.

This is a quick overview of the solution, not a comprehensive tutorial.

## Problem

Originally, PyMC4's proposed model specification API looked something like this:

<script src="https://gist.github.com/eigenfoo/8917e2fd72ea8940d54916c1cfbe1755.js"></script>

The main drawback to this API was that the `yield` keyword was confusing. Many users
don’t really understand Python generators, and those who do might only understand
`yield` as a drop-in replacement for return (that is, they might understand what it
means for a function to end in `yield` foo, but would be uncomfortable with bar =
`yield` foo).

Furthermore, the `yield` keyword introduces a leaky abstraction: users don’t care about
whether model is a function or a generator, and they shouldn’t need to. More generally,
users shouldn’t have to know anything about how PyMC works in order to use it: ideally,
the only thing users would need to think about would be their data and their model.
Having to graft several `yield` keywords into their code is a fairly big intrusion in
that respect.[^1]

Finally, this model specification API is essentially moving the problem off of our
plates and onto our users. The entire point of the PyMC project is to provide a friendly
and easy-to-use interface for Bayesian modeling.

To enumerate the problem,

1. Hide the yield keyword from the user-facing model specification API.
1. Obtain the user-defined model as a generator.
   * The main difficulty with the first goal is that as soon as we remove yield from the
     model function, it is no longer a generator.
   * But the PyMC core engine needs the model as a generator, since this allows us to
     interrupt the model at various points to do various things:
     + Manage random variable names.
     + Perform sampling.
     + Other arbitrary PyMC magic that I’m truthfully not familiar with.
   * In short, the user writes their model as a function, but we require the model as a
     generator.

Suffice to say, this problem isn't easy: https://github.com/eigenfoo/random/tree/master/python/ast-hiding-yield/00-prototype#why-is-this-problem-hard

## Solution

First, the `FunctionToGenerator` class:

<script src="https://gist.github.com/eigenfoo/43282c4e69156647d7bb2505f1dbafb2.js"></script>

This is the [recommended way of modifying
ASTs](https://greentreesnakes.readthedocs.io/en/latest/manipulating.html#modifying-the-tree),
and it's pretty well described by the docstring: the `visit_Assign` method adds the
`yield` keyword to all assignments by wrapping the visited `Assign` node within a
`Yield` node. The `visit_FunctionDef` method removes the decorator and renames the
function to `_pm_compiled_model_generator`.

Second, the `Model` class:

<script src="https://gist.github.com/eigenfoo/5e69bba2ab7ec6d6a2f53c13cbfd7b48.js"></script>

This class isn't meant to be instantiated: rather, it's [meant to be used as a Python
decorator](https://realpython.com/primer-on-python-decorators/#classes-as-decorators).
Essentially, it "uncompiles" the function to get the Python source code of the function.
This source code is then passed to the `parse_snippet`[^2] function, which returns the
AST for the function. We then modify this AST with the `FunctionToGenerator` class that
we defined above. Finally, we recompile this AST and execute it. Recall that executing
this recompiled AST defines a new function called `_pm_compiled_model_generator`. This
new function, accessed via the `locals()` variable[^3], is then bound to the class's
`self.model_generator`, which explains the confusing-looking line 25. This will be
useful later on.

Finally, the user facing API looks like this:

<script src="https://gist.github.com/eigenfoo/0bc4047ea0245a3f5f3c3a6ff8143154.js"></script>

As you can see, calling the `model_generator` method of `linear_regression` produces a
generator called `_pm_compiled_model_generator`, as desired. Success!

## Further reading

I've only given a high-level overview of this project here, and a lot of the technical
details were glossed over. If you're hungry for more, check out the following resources:

- Notebooks and more extensive documentation on this project [are on
  GitHub](https://github.com/eigenfoo/random/tree/master/python/ast-hiding-yield). In
  particular, it might be helpful to peruse the [links and references at the end of the
  READMEs](https://github.com/eigenfoo/random/tree/master/python/ast-hiding-yield/00-prototype#links-and-references).
- For those looking to programmatically inspect/modify Python ASTs the same way I did
  here, you might find [this Twitter
  thread](https://twitter.com/remilouf/status/1213079103156424704) helpful.
- And for those wondering how PyMC4's model specification API ended up, some very smart
  people gave their feedback on this work [on
  Twitter](https://twitter.com/avibryant/status/1150827954319982592).

---

[^1]: I was [subsequently convinced](https://twitter.com/avibryant/status/1150827954319982592) that this isn't a leaky abstraction after all.

[^2]: I omitted the the implementation of `parse_snippet` for brevity. If you want to see it, check out the "AST Helper Functions" section of this notebook: https://github.com/eigenfoo/random/blob/master/python/ast-hiding-yield/00-prototype/hiding-yield.ipynb

[^3]: For way more information on `exec`, `eval`, `locals()` and `globals()`, check out [Armin Ronacher's blog post](https://lucumr.pocoo.org/2011/2/1/exec-in-python/) and [this StackOverflow answer](https://stackoverflow.com/questions/2220699/whats-the-difference-between-eval-exec-and-compile).

