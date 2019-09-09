---
title: Python Packaging Cookbook
excerpt:
tags:
  - python
  - cookbook
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background12.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
toc: true
toc_sticky: true
toc_label: "Recipes"
toc_icon: "utensils"
last_modified_at: 2019-09-10
---

This is a cookbook, not a tutorial.

For personal projects.

## Checklist

You should have the following files in your top-level directory:

- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] `LICENSE`

## `__init__.py` (top-level)

The top-level `__init__.py` should contain some dunder variables that are
relevant to your package (e.g. `__version__`). You can write your `setup.py`
to grep out those variables.

## `__init__.py` (all others)

## `setup.py`

Just use [this template `setup.py`
file](https://gist.github.com/eigenfoo/8b644ca6518041e1742d1a7ca5739266).

<script src="https://gist.github.com/eigenfoo/8b644ca6518041e1742d1a7ca5739266.js"></script>

## Uploading to PyPI

Test PyPI exists!

You need an account on PyPI: you will be prompted for credentials.

```bash
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```

https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives

## Turning it into a command line script

## What is a `MANIFEST.in`?

## References

- https://packaging.python.org/tutorials/packaging-projects/
- https://python-packaging.readthedocs.io/en/latest/index.html
- https://setuptools.readthedocs.io/en/latest/setuptools.html

