---
title: "Why Topic Modelling Sucks"
excerpt: "When and why LDA might not be a great idea"
header:
  teaser: 
  overlay_image: 
  caption: ""
  last_modified_at: 2018-03-06
---

As I learn more and more about data science and machine learning, I've noticed
that a lot of resources out there go something like this:

> Check out this thing! It's great at this task! The important task! The one
> that was impossible/hard to do before! Look how well it does! So good! So
> fast!
> 
> It's dangerous to go alone! Take this! It's our tool/code/paper! We used it!
> To do the thing! And now you can use the thing too!

Joking and jeering aside, I do think its true that a lot of research and
resources are focused on what things _can_ do, or what things are _good_ at
doing. Invariably, when I actually implement the hyped-up "thing", I'd always be
frustrated when I didn't get as good results as the original paper.

Maybe I'm not smart enough to see this, but after I learn about a new technique
or tool or model, it's not immediately obvious to me when _not_ to use it. I
think it would be very helpful to learn what things _aren't_ good at doing, or
why things just plain _suck_ at times. Doing so not only helps you understand
the technique/tool/model better, but also sharpens your understanding of your
use case and the task at hand: what is it about your application that makes it
unsuitable for such a technique?

Which is why I'm writing the first of what will hopefully be a series of posts
about _"Why __ Sucks"_. I'll be outlining what

---

## Topic Modelling

First up: topic modelling. Specifically, [latent Dirichlet
allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), or LDA
for short (not to be confused with the other LDA, which I wrote a blog post
about before).

For those of you who've already encountered LDA and have seen [plate
notation](https://en.wikipedia.org/wiki/Plate_notation) before, this picture
will probably refresh your memory:

<img style="float: middle" width="500" height="500"
src="https://upload.wikimedia.org/wikipedia/commons/4/4d/Smoothed_LDA.png">

If you don't know what LDA is, fret not, for there is
[no](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
[shortage](http://obphio.us/pdfs/lda_tutorial.pdf)
[of](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/)
[resources](https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html)
[about](http://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation)
[this](https://radimrehurek.com/gensim/models/ldamodel.html)
[stuff](https://www.quora.com/What-is-a-good-explanation-of-Latent-Dirichlet-Allocation).
I'm going to move on to when and why LDA isn't the best idea.

**tl;dr:** _LDA and topic modelling doesn't work well with a) short documents,
in which there isn't much text to model, or b) documents that don't coherently
discuss a single topic._

Wait, wtf? Did George just say that topic modelling sucks when there's not much
topic, and not much text to model? Isn't that obvious?

_**Yes!** Exactly!_ Of course it's [obvious in
retrospect](https://en.wikipedia.org/wiki/Egg_of_Columbus)! Which is why I was
so upset when I realized I spent two whole weeks faffing around with LDA when
topic models were the opposite of what I needed, and so frustrated that more
people aren't talking about when _not_ to use/do certain things.

But anyways, `<\rant>` and let's move on to why I say what I'm saying.

Recently, I've taken up a project in investigating and modelling hate speech on
Reddit. There are, of course, many ways one can take this project, but something
I was interested in was finding similarities between subreddits, clustering
comments, and visualizing these clusters somehow. Naturally, I turned to topic
modelling and dimensionality reduction.

The techniques that I came across first were LDA ([latent Dirichlet
allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)) and
t-SNE ([t-distributed stochastic neighbor
embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)).
I can't say that its a popular choice of two techniques, but tehre have been
some successes. For instance, Shuai had some success with this method [when
using it the 20 News Groups
dataset](https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html);
some work done by Kagglers have [yielded reasonable
results](https://www.kaggle.com/ykhorramz/lda-and-t-sne-interactive-visualization),
and [the StackExchange community doesn't think its a crap
idea](https://stats.stackexchange.com/questions/305356/plot-latent-dirichlet-allocation-output-using-t-sne).

