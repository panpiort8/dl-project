# dl-project

## Overview

Reproduces and extends [Pretrained Transformers as Universal Computation Engines](https://arxiv.org/abs/2103.05247).

### Pipeline

Open our pipeline in [Colab](https://colab.research.google.com/github/panpiort8/dl-project/blob/master/pipeline.ipynb).

### Datasets

#### MNIST

//TODO short dataset description

#### CIFAR 10

//TODO short dataset description

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

### Baselines

//TODO

Dataset | Metric Name | Result | #runs
:---: | :---: | :---: | :---:
MNIST | Accuracy | ... | ...
CIFAR 10 | Accuracy | ... | ...
MNIST Digits Addition | ? | ... | ...
Cyp3A4 Inhibition | Accuracy | ... | ...
Speech Command | Accuracy | ... | ...

#### MNIST

//TODO what is the baseline? Short description and reference.

#### CIFAR 10

//TODO what is the baseline? Short description and reference.

#### MNIST Digits Addition

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

