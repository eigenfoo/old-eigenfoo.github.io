---
title: Data Collection is Hard. You Should Try It.
excerpt: "No, seriously."
tags:
  - data
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background9.png
  overlay_filter: 0.1
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2022-03-03
---

For people who make careers out of data, data scientists don't have *nearly*
enough experience in data collection, and many data scientists don’t even seem
to feel the need to develop experience collecting data.

Puzzlingly, this trend doesn’t seem to be true of other forms of unglamorous
data work like data cleaning (where people generally accept that [data cleaning
is not grunt
work](https://counting.substack.com/p/data-cleaning-is-analysis-not-grunt)).

With this blog post I want to give a defense of data collection — not as an
activity that’s inherently worthwhile pursuing (I assume data scientists don’t
need to be convinced of that!), but as something that is worth doing even for
*selfish* reasons. Why should you spend time learning about that data
collection system that's being maintained by that other team at work? Why
should you consider collecting some data for your next side project? What's in
it for _you_?

Throughout this blog post, I’ll be making comparisons to a recent project of
mine, [`cryptics.georgeho.org`](https://cryptics.georgeho.org/), a dataset of
cryptic crossword clues.

## Learn Data-Adjacent Technologies

The most obvious reason is that data collection is a fantastic opportunity to
familiarize yourself with many staple technologies in data - and there aren't
that many side projects that run the entire data tech stack!

To enumerate:

- Compute services
    - Your data collection pipelines will obviously need to run somewhere. Will
      that be in the cloud, or on your local computer? How do you think about
      trading off cost, compute and convenience?
    - I ran most of my web scraping on DigitalOcean Droplets, but I could just
      as easily have taken the opportunity to learn more about cloud compute
      solutions or serverless functions like AWS EC2 or Lambda. These days, the
      project runs incremental scrapes entirely on my laptop.
- Data storage
    - You’ll need to store your data somewhere, whether it be a relational or
      NoSQL database, or just flat files. Since your data will outlive any code
      you write, careful design of the data storage solution and schema will
      pay dividends in the long run.
    - I used SQLite for its simplicity and performance. However, as the scope
      of the project expanded, I had to redesign the schema multiple times,
      which was painful.
- Labeling, annotation or other data transformations
    - After collecting your data, you may want to label, annotate or other
      structure or transform your data. For example, perhaps you’ll want to
      pull structured tabular data out of unstructured PDFs or HTML tag soups;
      another example might be to have a human label the data.
    - This is the main “value-add” of your dataset — while the time and effort
      required to collect and store the data constitutes a moat, ultimately
      what will distinguish your dataset to *users* will be the transformations
      done here.
    - For me, this involved a lot of `BeautifulSoup` to parse structured data
      out of HTML pages. This required a [significant amount of development and
      engineering
      effort](https://cryptics.georgeho.org/datasheet#collection-process). 
- Data licensing and copyright
    - Once you have your dataset, what is the legality of licensing, sharing or
      even selling your data? The legality of data are a huge grey area
      (especially if there’s any web scraping involved), and while navigating
      these waters will be tricky, it’s instructive to learn about it. 
    - I feel like the collection and structuring of cryptic crossword clues for
      academic/archival purposes was fair use, and so didn’t worry too much
      about the legality of my project — but it was an educational rabbit hole
      to fall down!
- Sharing and publishing data
    - The legal nuances of data aside, the technical problem of sharing data is
      pretty tricky!
    - This problem sits at the intersection of MLOps and information design:
      you want to share the data in a standardized way, while having an
      interface that making it easy for users to explore your data. Serving a
      tarball on a web server technically works, but leaves so much on the
      table.
    - `cryptics.georgeho.org` uses [Datasette](https://datasette.io/), which I
      can’t recommend highly enough.
- Writing documentation
    - If you think it’s hard to write and maintain good documentation for
      software, imagine how difficult it must be to do the same for data, which
      outlives software and is much harder to both create and version control.
    - I’ve found [Gebru et al.’s Datasheets for
      Datasets](https://arxiv.org/abs/1803.09010) to be an excellent template
      for documenting data.

## Design a Data Collection System

More importantly, starting a small data collection project is a great way to
get experience designing an entire data pipeline from end to end. This kind of
opportunity doesn't come easily (even in industry!), and while your data
pipeline won't be as sophisticated as the kinds you'll find in data companies,
you'll be be able to take away some valuable lessons from it.

