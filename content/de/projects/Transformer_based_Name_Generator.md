---
title: "Transformer based Name Generator"
date: 2022-04-26T00:00:00Z
draft: false
---

This is the first of a three part blog post series in which we are going to develop a 
Transformer based name generator

* 1. Exploring the model and generating the first names
* 2. Evaluate the quality of the generated names
* 3. Perform hyper parameter tuning to better understand the model architecture and improve data quality

## Overview and Introduction
In this blog post series you will learn how to build a pytorch model that learns
to generate new names from a list of examples.

The machine learning model will be based on the very successful and widely applied
transformer architecture.
For a excellent introduction to transformers and self attention as their key
idea have a look a [this blog post by Peter Bloem](http://peterbloem.nl/blog/transformers).

Once we have trained our model and used it to generate a set of names, we are going
to focus on evaluating the generated names, discussing what we want to achieve
with a generative model and how to quantify the data quality.

Once we have some good data quality metrics, we shift our focus on hyperparameter
tuning with [wandb](https://docs.wandb.ai/quickstart) to improve our results
and in addition get a better understanding of our model.

I hope you are looking forward to this series and find it useful and inspiring.

## Project Repository and System configuration
All the code for this tutorial series is available on github as the
[onomatico](https://github.com/mapa17/onomatico) project.

The project makes use of [poetry](https://python-poetry.org/docs/basic-usage/) 
as the dependency management system, and I as always recommend to work with your
python projects in a virtual environment that is build with conda + pip.

If working locally is not an option for you, or you want to make use of the power
of the cloud, I recommend to check out an previous blog post that helps you to
[setup your ML development environment in the cloud in less than 5 min](ML_dev_deployment_on_AWS.md).

## Lets start with the beginning, the training data
The goal of this tutorial is to build a generative model that is capable of creating
american name pairs of first and last names. As all machine learning models we
need training data to teach the model based on positive examples what "good" names
look like.

The training data we are going to use are build based on a set of the 100 most
common american first and last names available [here](https://github.com/fivethirtyeight/data/tree/master/most-common-name).

To be more precise, I have taken the 100 most common last names, and have combined
them with 10 random picked first names from the 100 most common first names.

The result is a data set of 1000 name combinations from which we will take 900
to train our generative model and 100 to validate it.




## Project Structure and Components
The project is fully contained in github repo [onomatico](https://github.com/mapa17/onomatico) (for the curious, the name was inspired by a wordplay on [onomatics](https://en.wikipedia.org/wiki/Onomastics), the study of names.).

The repo contains 
* **main folder**: several configuration files to setup the project and configure additional tools (poetry, wandb) 
* **data folder**: the training and test data used in this blog post
* **onomatico**: the python modules tha are used to train a model and generate new names

Pricing information: https://instances.vantage.sh/?region=eu-central-1&compare_on=true&selected=g3s.xlarge

