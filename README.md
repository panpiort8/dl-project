# dl-project

## Overview

Reproduces and extends [Pretrained Transformers as Universal Computation Engines](https://arxiv.org/abs/2103.05247).

## Pipeline

Open our pipeline in [Colab](https://colab.research.google.com/github/panpiort8/dl-project/blob/master/pipeline.ipynb).


### Weights and Biases

Check our results in [wandb](https://wandb.ai/dl-project2) page.

### Datasets

#### MNIST

The MNIST database contains 60,000 training images and 10,000 testing images of handwritten digits.
We use the standard MNIST benchmark, where the model must classify 32 × 32 black-and-white image. The tokens given to the model are 4 × 4 image patches, so
the models are fed 64 tokens of dimension 16.

#### CIFAR 10

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.
We use the standard CIFAR-10 benchmark, where the tokens given to the model are 4 × 4 image patches, so the models are fed 64 tokens of dimension 16.

#### MNIST Digits Addition

//TODO short dataset description

#### Speech Commands

Speech Commands dataset contains short audio clips of a fixed number of command words such as “stop”, “go”, “up”,
“down”, etc spoken by many speakers. Google released two versions of the dataset with the first version containing 65k
samples over 30 classes, and the second containing 110k samples over 35 classes. In our project, we used the second
version which is available in the [torchaudio](https://pytorch.org/audio/stable/index.html) library.

#### Cyp3A4 Inhibition


The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within
cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It
oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the
body.

#### Baselines

//TODO

Dataset | Metric Name | Result | #runs
:---: | :---: | :---: | :---:
MNIST | Accuracy | 99.5% | ...
CIFAR 10 | Accuracy | 73.6% | ...
MNIST Digits Addition | ? | ... | ...
Cyp3A4 Inhibition | Accuracy | ... | ...
Speech Command | Accuracy | 98.1%  | ...

#### MNIST

The baseline for this dataset is LSTM, taken from original [paper](https://arxiv.org/abs/2103.05247).

#### CIFAR 10

The baseline for this dataset is LSTM, taken from original [paper](https://arxiv.org/abs/2103.05247).

#### MNIST Digits Addition

Modele get presented sequence of `n` MNIST digits (28x28 pixels) and should predict sum of them. Task is parametrized by sequence length (for `n=1` it is equivalend to standard MNIST task).

Source: Neural Arithmetic Logic Units (https://arxiv.org/pdf/1808.00508.pdf)


//TODO what is the baseline? Short description and reference.


#### Cyp3A4 Inhibition

The baseline for this dataset is Molecule Attention Transformer, fine-tuned
with [huggingmolecules](https://github.com/gmum/huggingmolecules) package (on the same data split, with a default hps
setting).

#### Speech Commands

The baseline for this dataset is Audio Spectrogram Transformer, more details can be found
in [paper](https://arxiv.org/abs/2104.01778).

## Question 1

Can pretrained language models transfer to different modalities?

### Methodology

1. Train FPT on all datasets with default parameters set:

```python
experiments_params = dict(
    # ...
    model_name='gpt2',
    pretrained=True,

    freeze_trans=True,
    freeze_in=False,
    freeze_pos=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_out=False,
    # ...
)
```

2. Compare the results with the results of [baselines](#Baselines). Are they somehow comparable?

### Empirical results

Result = average of test accuracy on k steps (for all experiments k=100)

Dataset | Metric Name | Result | #runs | #steps | Parameters
:---: | :---: | :---: | :---: | :---: | :---:
MNIST | Accuracy | 98.15% | 1 | 250 | steps_per_iter=200 <br /> test_steps_per_iter=100  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=4
CIFAR10 | Accuracy | 63.24% | 1 | 550 | steps_per_iter=200 <br /> test_steps_per_iter=100  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=4
MNIST Digits Addition | ? | ... | ... | ...
Cyp3A4 Inhibition | Accuracy | 75% | 1 | 400 | steps_per_iter=100 <br /> test_steps_per_iter=25  <br /> learning_rate=1e-3  <br /> batch_size=16
Speech Command | Accuracy | 8.69% | 1 | 400 | steps_per_iter=200 <br /> test_steps_per_iter=50  <br /> learning_rate=1e-3  <br /> batch_size=16

### Conclusions

* Our preliminary results supports author's thesis.
* Comparison with baselines:

Model | MNIST | CIFAR10 | MNIST Digits Addition | Cyp3A4 Inhibition | Speech Command
:---: | :---: | :---: | :---: | :---: | :---:
FPT | 98.15% | 63.24% | ... | 75% | 8.69%
Baseline | 99.5% | 73.6% | ... | ... | ... 

## Question 2

What is the importance of the pretraining modality?

### Methodology

1. Train FPT on all datasets without the pretraining and freezing:

```python
experiments_params = dict(
    # ...
    model_name='gpt2',
    pretrained=False,

    freeze_trans=False,
    freeze_in=False,
    freeze_pos=False,
    freeze_ln=False,
    freeze_attn=False,
    freeze_ff=False,
    freeze_out=False,
    # ...
)
```

2. Compare the results with the results of [baselines](#Baselines) and the results from [question 1](#Question 1). Are
   they somehow comparable?

### Empirical results
//TODO Result = average of n runs

Dataset | Metric Name | Result | #runs
:---: | :---: | :---: | :---:
MNIST | Accuracy | ... | ...
CIFAR 10 | Accuracy | ... | ...
MNIST Digits Addition | ? | ... | ...
Cyp3A4 Inhibition | Accuracy | ... | ...
Speech Command | Accuracy | ... | ...

### Conclusions

// TODO

## Question 3

Does the transformer architecture provide inductive bias that transfers well to various modalities?

### Methodology

1. Train FPT on all datasets without the pretraining, but with freezing:

```python
experiments_params = dict(
    # ...
    model_name='gpt2',
    pretrained=False,

    freeze_trans=True,
    freeze_in=False,
    freeze_pos=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_out=False,
    # ...
)
```

2. Compare the results with the results of [baselines](#Baselines) and the results from [question 1](#Question 1)
   and [question 2](#Question 2). Are they somehow comparable?

### Empirical results

//TODO Result = average of n runs

Dataset | Metric Name | Result | #runs
:---: | :---: | :---: | :---:
MNIST | Accuracy | ... | \>=3
CIFAR 10 | Accuracy | ... | \>=3
MNIST Digits Addition | ? | ... | \>=3
Cyp3A4 Inhibition | Accuracy | ... | \>=3
Speech Command | Accuracy | ... | \>=3

### Conclusions

// TODO

## Question 4

Can pretrained visual models transfer to different modalities?

### Methodology

0. Implement using ViT as the pretrained transformer.
1. Train FPT on all datasets with default parameters set, but with ViT as pretrained transformer:

```python
experiments_params = dict(
    # ...
    model_name='vit',
    pretrained=True,

    freeze_trans=True,
    freeze_in=False,
    freeze_pos=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_out=False,
    # ...
)
```

2. Compare the results with the results of [baselines](#Baselines) and the results from [question 1](#Question 1). Are they somehow comparable?

### Empirical results

//TODO Result = average of n runs

Dataset | Metric Name | Result | #runs
:---: | :---: | :---: | :---:
MNIST | Accuracy | ... | ...
CIFAR 10 | Accuracy | ... | ...
MNIST Digits Addition | ? | ... | ...
Cyp3A4 Inhibition | Accuracy | ... | ...
Speech Command | Accuracy | ... | ...

### Conclusions

// TODO
