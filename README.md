# dl-project

## Overview

Reproduces and extends [Pretrained Transformers as Universal Computation Engines](https://arxiv.org/abs/2103.05247).

## Pipeline

Open our pipeline in [Colab](https://colab.research.google.com/github/panpiort8/dl-project/blob/master/pipeline.ipynb).

## Datasets

### Cyp3A4 Inhibition

The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within
cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It
oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the
body.

#### Baselines

Dataset | Accuracy
:---: | :---:
Cyp3A4 Inhibition | ...

#### Cyp3A4 Inhibition

The baseline for this dataset is Molecule Attention Transformer, fine-tuned with [huggingmolecules](https://github.com/gmum/huggingmolecules) package (on the same data split, with a default hps setting).

### MNISTDigitAddition

Modele get presented sequence of `n` MNIST digits (28x28 pixels) and should predict sum of them. Task is parametrized by sequence length (for `n=1` it is equivalend to standard MNIST task).

Source: Neural Arithmetic Logic Units (https://arxiv.org/pdf/1808.00508.pdf)


#### Results ####

For n=2, after 50x100 training samples
```
=========================================================
| Iteration                 |                        50 |
| Gradient Steps            |                      5000 |
| Average Train Loss        |       0.18916434701532125 |
| Start Train Loss          |        0.1235654279589653 |
| Final Train Loss          |       0.24141760170459747 |
| Test Loss                 |       0.22688765376806255 |
| Test Accuracy             |        0.9516000000000001 |
| Train Accuracy            |        0.9496000000000002 |
| Time Training             |         5.026964902877808 |
| Time Testing              |        0.7528393268585205 |
```

For n=10, after 50x100 training samples

```
=========================================================
| Iteration                 |                        50 |
| Gradient Steps            |                      5000 |
| Average Train Loss        |        2.1147920048236846 |
| Start Train Loss          |         2.085275888442993 |
| Final Train Loss          |        2.1533188819885254 |
| Test Loss                 |         2.261877069473267 |
| Test Accuracy             |       0.23879999999999998 |
| Train Accuracy            |                    0.2103 |
| Time Training             |         15.22073483467102 |
| Time Testing              |        2.8992207050323486 |
```

TODO: calculate MSE

#### Dummy model ####
```
n=1 -> 0.1
n=2 -> 0.09913
n=10 -> 0.04434
n=100 -> 0.01444
n=1000 -> 0.0047
```

#### BaseLines

TBA: https://arxiv.org/pdf/1808.00508.pdf

