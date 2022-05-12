---
title: "Transformer based Name Generator - Part 1"
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
as the dependency management system, and I as always, recommend to you, to configure
your python projects in a virtual environment based on conda + pip.

If you dont have a local working environment, or you want to make use of the power
of the cloud, I recommend to check out an previous blog post of mine that helps you to
[setup your ML development environment in the cloud in less than 5 min](ML_dev_deployment_on_AWS.md).

## Lets start at the beginning, the training data
The goal of this tutorial is to build a generative model that is capable of creating
american name pairs of first and last names. As all machine learning models we
need training data to teach the model based on positive examples what "good" names
look like.

The training data we are going to use is build based on a set of the 100 most
common american first and last names available [here](https://github.com/fivethirtyeight/data/tree/master/most-common-name).

To be more precise, it contains the 100 most common last names, and combines
them with 10 random picked first names from the 100 most common first names.

The result is a data set of 1000 name combinations from which we will take 900
to train our generative model and 100 to guide the training process.

## Project Structure and Components
Once you cloned the project repo [onomatico](https://github.com/mapa17/onomatico) (for the curious, the name was inspired by a wordplay on [onomatics](https://en.wikipedia.org/wiki/Onomastics), the study of names.) you can find the following
project structure and sub-folders containing

* **main folder**: several configuration files to setup the project and configure additional tools (poetry, wandb) 
* **data folder**: the training and test data used in this blog post
* **onomatico**: the python modules tha are used to train a model and generate new names
* **deployment**: Terraform configuration and system configuration files to launch AWS instances (fore more details [see](ML_dev_deployment_on_AWS.md) 

For the rest of this tutorial we will focus on the `onomatico` folder that contains `onomatico/main.py`
which provides the frontend in the form of an CLI application and `onomatico/utils`
that provides two python modules to help access and abstract the training data `onomatico/utils/Names.py`
and defines our model `onomatico/utils/Transformer.py`.

We start with the last two utility modules, explaining first the Transformer model and than how to access the training data.

## A Transformer based generative character model
The code used to build this model is derived from the official [Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
and contains the main Transformer class that is building a Transformer model
on sequences of individual characters.

The class constructor, combines multiple [TransformerEncoderLayers](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
(which make up the heart of the Transformer) preceded by an Embedding layer
that learns distributed representations of our tokens and a positional encoding function
that modulates the embedded tokens to retain information about their position
in the input sequence. At the output of the Transformer we have a linear layer
that maps to our set of tokens, which in our case are individual characters.

An overview of the Transformer architecture you can see in the following diagram
(taken form [this blog post](https://charon.me/posts/pytorch/pytorch_seq2seq_6/))
![Transformer Architecture](images/TransformerArchitecture.jpg)

Lets have a look at `onomatico/utils/Transformer.py` in a bit more detail

```python
import math
import inspect
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    """Add absolute position encodings to embedding vectors.

    For each embedding token add an (additative) position dependent term that
    is calculated as an superposition of sinus and cosines functions that is
    unique for each position in a sequence.

    For a simple visual instruction watch: https://youtu.be/dichIcUZfOw?t=318

    Copied from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#load-and-batch-data
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    kwargs: Dict[str, None]

    def __init__(
        self,
        ntokens: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.2,
    ):
        """Transformer based generative model learning to reproduce token sequences.

        Args:
            ntokens (int): Size of the vocab used for training and generation
            d_model (int): Embedding size of the input and output of the Feed Forward block of a single Transformer Layer
            nhead (int): number of self attention heads
            d_hid (int): Internal Embedding size of the Feed Forward block (emb sizes: d_model x d_hid x d_model)
            nlayers (int): Number of Transformer Layers
            dropout (float, optional): Dropout rate used for the Multi-headed attention and Feed Forward block. Defaults to 0.5.
        """
        super().__init__()

        # Store the kwargs as a dict, so they can be saved with the model
        # and reused when loading the model.
        s = inspect.signature(self.__init__)
        l = locals()
        self.kwargs = {k: l[k] for k in s.parameters.keys()}

        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Use the defined Model components and perform the actual computation.

        As input we expect an tensor that contains multiple input sequences (all of 
        the same length) and a mask tensor that indicates for each position in
        the sequence what other positions can be used  in the self attention mechanism. 

        The output are raw, unnormalized scores (logits) for each token position
        that have to be fed to an activation function (e.g softmax) in order to
        be interpreted as a probability distribution of the vocabulary.

        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

```

For a discussion about the architecture of Transformers I can recommend
[this](https://e2eml.school/transformers.html)
and [the following](https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3) reference.

With the hyperparameter that are provided during the constructor of the class we
can change the architecture of our model by for example increasing the size of
the internal embedding layers used in `TransformerEncoderLayers`, the number of
heads or the number of layers themselves. In addition we are specifying the
number of tokens in our vocabulary for the model to be able to read all inputs
and generate names containing only characters that are present in our vocabulary.

Finding the best hyper parameters for our model is a challenging task, and we will
discuss it later in detail, but for now, as a quick peek about what is to come,
we will see that increasing the embedding sizes has little positive effect on
data quality, but increasing the number heads does.

## Loading and Preprocessing of the dataset
The dataset is a csv file containing in a single column the first and last name.
Whereby the last name is spelled on purpose in all capital letters.

Example
```csv
name
Jeffrey SMITH
Amanda SMITH
Justin SMITH
Michelle SMITH
Jennifer SMITH
Kathleen SMITH
Linda SMITH
Larry SMITH
Jacob SMITH
...
```

The idea behind the special writing of the Last Name in only capital letters, is
to include certain amount of structural information that our generative
model needs to learn to reproduce, in addition to the individual character distribution.

The module `onomatico/utils/Names.py` contains two classes that help with loading
the data and creating a pytorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) (i.e `NamesDataset`)
and another class (i.e. `Names`) that we use to iterate over the dataset
during training to provide us with pairs of training data (i.e x, y) in mini-batches.

```python
class Names(Iterator):
    """Wraps a `NamesDataset` providing an iterator that is used during training, returning
    mini batches of data points and raising StopIteration after one iteration over
    the complete dataset (i.e. epoch).
    """
    names_dataset: NamesDataset
    names_dl: DataLoader
    names_iter: Iterator
    device: torch.device
    batch_size: int

    def __init__(self, csv_file: Path, batch_size: int, device: torch.device, **param):
        """Build a torch DataLoader reading from given csv_file

        Args:
            csv_file (Path): Path to csv file containing names
            batch_size (int): Batch size
            device (torch.device): torch device
        """
        self.names_dataset = NamesDataset(csv_file, **param)
        self.names_dl = DataLoader(
            self.names_dataset, batch_size=batch_size, shuffle=True
        )
        self.names_iter = iter(self.names_dl)
        self.batch_size = batch_size

    def get_padded_sequence_length(self) -> int:
        return self.names_dataset.padded_sequence_length

    def get_batch(self) -> Tuple[torch.tensor, torch.tensor]:
        """Returns a mini-batch of names, with the target is shifted by one position.

        Returns:
            Tuple[torch.tensor, torch.tensor]: Mini-batch of (training, target)
        """
        # Get a mini-batch of encoded name token sequences
        batch = next(self.names_iter)

        # The target is shifted by one element
        # data: "<Bob MILLER>!!",
        # target: "Bob MILLER>!!!"
        data = batch[:, :-1]
        target = batch[:, 1:]
        return data, target

    def __len__(self) -> int:
        return len(self.names_dl)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Tuple[torch.tensor, torch.tensor]:
        """Wrap the iterator generated from the DataLoader.
        Make sure to get refresh the iterator once it is emptied.
        So for a single epoch we raise a StopIteration exception, but can read
        from the Dataset again in the next epoch.

        Raises:
            e: StopIteration

        Returns:
            Tuple[torch.tensor, torch.tensor]: Training and Label data 
        """
        try:
            return self.get_batch()
        except StopIteration as e:
            self.names_iter = iter(self.names_dl)
            raise e
```

Because of the small size of the training data, we load the complete data set into
memory and transform it before training (see `__create_vocab_and_tokens()`).
Creating a list of token sequences of the same length, surrounding each name with
a `start` (`<`), an `end` (`>`) and a padding (`!`) token.

Example: `Bob MILLER -> <Bob MILLER>!!!`

The start token will be used during the generation of new names as the `seed`
symbol for our model and based on the end token we can identify when the model
'thinks' that it has completed it job and finished generating a name.

The padding token is used to have the same sequence length for each name, so we
can batch multiple names together in a tensor of the same size.

## CLI Interface
Part of the project is a CLI interface (`onomatico/main.py`) that has the following sub commands that can be used to:
* `vocab`: Create a vocab out of CSV training data, that is needed for model training and generation of new names.
* `train`: Train a model using a vocab and training data.
* `generate`: Generate new names using the trained model and a vocab.
* `compare`: Compare the similarity between original and generated names.

You can explore the arguments required for each command with

`onomatico --help`

The CLI is build making use of the [Typer](https://typer.tiangolo.com/) library that is a derivation of [Click](https://click.palletsprojects.com/en/8.1.x/) to reduce the boiler plait code make use of the docstrings and type hints and perform a simple type input validation when handling the program arguments.

## Train a model and generate some names
Its time to install the project dependencies, create some working directory for temporary files, train a model and generate some names.

Inside the project directory and your virtual environment run something like the following

```bash
poetry install
mkdir WD
onomatico vocab data/names.csv WD/vocab.pt
onomatico train data WD/vocab.pt 50 WD/model.pt --disable-wandb
onomatico generate WD/model.pt WD/vocab.pt 900 WD/new_names.csv
cat WD/new_names.csv
```

Congratulations you have created a set of 900 new names. Lets have a peek at
them quickly.

```bash
names
Joshan MARTIS
Christie TARE
Zachar KEKS
Chttew GOJEZ
Limon DARTISON
Nichathelld GUEZ
Chelles RARD
Edonda GERTIZSON
Chrim SCOTTEZ
Aaridony MOWNEZ
Chonarl TUGUE
Michele MUNERES
Ronald JAYAN
Jotthhan PERTERE
...
```

Not to bad, but one can spot already some issues with the names, but this
will bring is to the topic of data quality that will be covered in the
next blog post.

So stay tuned for more!

## Conclusion
Congratulations for making it through the tutorial. You hopefully got an basic understanding on how to build a generative model and how to use the tutorial project
to generate new names.

But what about those new names? How do they compare with the training data?
Are they simple copies or shuffles of the original names? Do the names make any
sense, or are they only random shuffles of characters?
How much similarity and how much novelty do we want in our new names, and how do
we quantify the similarity between them?

This and other questions we are going to discuss in the next blog post of this
series.
