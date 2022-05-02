---
title: "Transformer based Name Generator - Part 2"
date: 2022-05-02T00:00:00Z
draft: false
---
Welcome back to the second part of this blog post series about building a Transformer
based name generator.

If you missed the first part, you can find it [here](Transformer_based_Name_Generator_part1.md).

At the end of part 1 we generated a set of names and had a quick peek at those
names. In this part we are going to dive deeper into the topic of data quality,
how to define it, how to measure it and how to use it to tune our model.

## What do we really want?
Like in other areas in live, its not so easy to define what we want in detail and
our generative model is no exception.

What does a set of "good" names look like?

Of course it would make no sense if the model would generate random sequences of
characters (being one extreme of the spectrum) but at the same if, the model 
would be not very useful if it would simply memorize the training dataset, and
generate list of names that are mere repetitions of names in the training data
(i.e. the other extreme).

In the best case we want an [generative model](https://developers.google.com/machine-learning/gan/generative)
to exhibit a healthy balance between being knowledge able and being creative.

The model should *know* what names look like (i.e. be knowledge able about the domain in question)
and at the same time should be able to create *new* and *novel* names (i.e. be creative).

