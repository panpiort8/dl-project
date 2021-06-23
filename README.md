# dl-project

## Overview

Reproduces and extends [Pretrained Transformers as Universal Computation Engines](https://arxiv.org/abs/2103.05247).

## Pipeline

Open our pipeline in [Colab](https://colab.research.google.com/github/panpiort8/dl-project/blob/master/pipeline.ipynb).

### Weights and Biases

Check our results in [wandb](https://wandb.ai/dl-project2) page.

### Datasets

#### MNIST

The MNIST database contains 60,000 training images and 10,000 testing images of handwritten digits. We use the standard
MNIST benchmark, where the model must classify 32 × 32 black-and-white image. The tokens given to the model are 4 × 4
image patches, so the models are fed 64 tokens of dimension 16.

#### CIFAR 10

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent
airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. We use
the standard CIFAR-10 benchmark, where the tokens given to the model are 4 × 4 image patches, so the models are fed 64
tokens of dimension 16.

#### MNIST Digits Addition

Model is presented with a sequence of `n` MNIST digits (28x28 pixels) and should predict the sum of them. Task is
parametrized by a sequence length (for `n=1` it is equivalend to standard MNIST task). Task is taken from this [paper](https://arxiv.org/pdf/1808.00508.pdf).

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

    
#### TODO:
    
- Complete table?
- lepsze baseliny: https://paperswithcode.com/sota/image-classification-on-cifar-10
- określanie jaki jakie score powinien osiągać "random"


Dataset | Metric Name | Result 
:---: | :---: | :---: | 
MNIST | Accuracy | 99.5% | 
CIFAR 10 | Accuracy | 99.5% | 
MNIST Digits Addition (n=10) | Mean Absolute Error | 1.42 (dummy: 7.31) | 
Cyp3A4 Inhibition | Accuracy | 82.1% |
Speech Command | Accuracy | 98.1%  | 

#### MNIST

The baseline for this dataset is LSTM, taken from original [paper](https://arxiv.org/abs/2103.05247).

#### CIFAR 10

The baseline for this dataset is VIT-H, taken from this [paper](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1).

#### MNIST Digits Addition

The baseline for this dataset is NAC model from ["Neural Arithmetic Logic Units"](https://arxiv.org/pdf/1808.00508.pdf) paper. It is a model designed for addition / subtraction task, and it perform linear affine tranformation of its input. 

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

1. Train Frozen Pretrained Transformer (FPT) on all datasets with default parameters set:

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

Dataset | Metric Name | Result | #runs | #steps | Parameters | wandb
:---: | :---: | :---: | :---: | :---: | :---: | :---:
MNIST | Accuracy | 98.15% | 1 | 250 | steps_per_iter=200 <br /> test_steps_per_iter=100  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=4 | [3boh2kd2](https://wandb.ai/dl-project2/universal-computation-engine/runs/3boh2kd2)
CIFAR10 | Accuracy | 63.24% | 1 | 550 | steps_per_iter=200 <br /> test_steps_per_iter=100  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=4 | [3qo22alh](https://wandb.ai/dl-project2/universal-computation-engine/runs/3qo22alh)
MNIST Digits Addition | MSE | 7.404 | 1 | 200 | steps_per_iter=200 <br /> test_steps_per_iter=50  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=28 <br /> n=10 | [2mfojag2](https://wandb.ai/dl-project2/universal-computation-engine/runs/2mfojag2)
Cyp3A4 Inhibition | Accuracy | 75.65% | 1 | 280 | steps_per_iter=200 <br /> test_steps_per_iter=50  <br /> learning_rate=1e-3  <br /> batch_size=16 | [3kzurs4w](https://wandb.ai/dl-project2/universal-computation-engine/runs/3kzurs4w)
Speech Command | Accuracy | 64.97% | 1 | 500 | steps_per_iter=200 <br /> test_steps_per_iter=25  <br /> learning_rate=1e-4  <br /> batch_size=16 <br /> patch_size=80 | [3mi762gc](https://wandb.ai/dl-project2/universal-computation-engine/runs/3mi762gc)

### Conclusions

* Our preliminary results supports author's thesis.
* Comparison with baselines:

Model | MNIST | CIFAR10 | MNIST Digits Addition | Cyp3A4 Inhibition | Speech Command
:---: | :---: | :---: | :---: | :---: | :---:
FPT | 98.15% | 63.24% | 7.404* | 75.65% | 64.97%
Baseline | 99.5% | 99.5% | 1.42 | 82.1% | 98.1%

## Question 2

What is the importance of the pretraining modality?

### Methodology

1. Train Unfrozen Pretrained Transformer (UPT) on all datasets without the pretraining and freezing:

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



Dataset | Metric Name | # runs | mean | std | weights | Reference
:---: | :---: | :---: | :---: | :---: | :---: | :---:
MNIST | Accuracy | 6 | 60.76% | 0.1154 | 124,460,554 | [WandB](https://wandb.ai/dl-project2/universal-computation-engine/groups/mnist-unfreeze-mnist)
CIFAR 10 | Accuracy | 3 | 21.73% | 0.0156 | 124,485,130 | [WandB](https://wandb.ai/dl-project2/universal-computation-engine/groups/cifar10-unfreeze-cifar10)
MNIST Digits Addition (Regression, n=10) | Mean Absolute Error | 3 | 7.41 | 0.0316 | 125,043,457 | [WandB](https://wandb.ai/dl-project2/universal-computation-engine/groups/mnist-add-reg-unfreeze-mnist-add-reg)
MNIST Digits Addition (Classification, n=10) | Mean Absolute Error | 5 | 6.55 | 0.9497 | 125,112,667 | [WandB](https://wandb.ai/dl-project2/universal-computation-engine/groups/mnist-add-unfreeze-mnist-add)
Cyp3A4 Inhibition | Accuracy | 3 | 61.82% | 0.0308 | 125,209,346 | [WandB](https://wandb.ai/dl-project2/universal-computation-engine/groups/cyp3a4-unfreeze-cyp3a4)
Speech Command | Accuracy | 2 | 62.52% | - | 124,528,931 | [WandB](https://wandb.ai/dl-project2/universal-computation-engine/runs/1fao2s4o)

### Conclusions


Model | MNIST | CIFAR10 | MNIST Digits Addition | Cyp3A4 Inhibition | Speech Command
:---: | :---: | :---: | :---: | :---: | :---:
UPT | 60.76% | 21.73% | 7.41* | 61.82% | 63.56%
FPT | 98.15% | 63.24% | 7.404* | 75.65% | 64.97%
Baseline | 99.5% | 99.5% | 1.42 | 82.1% | 98.1%


## Question 3

Does the transformer architecture provide inductive bias that transfers well to various modalities?

### Methodology

1. Train Frozen Random Transformer (FRT) on all datasets without the pretraining, but with freezing:

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

Result = average of test accuracy on k steps (for all experiments k=100)

Dataset | Metric Name | Result | #runs | #steps | Parameters | wandb
:---: | :---: | :---: | :---: | :---: | :---: | :---:
MNIST | Accuracy | 97.32% <br /> 97.08% <br /> 96.76% | 3 | 350 <br /> 250 <br /> 220 | steps_per_iter=200 <br /> test_steps_per_iter=50  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=4 | [3v1wtr64](https://wandb.ai/dl-project2/universal-computation-engine/runs/3v1wtr64) <br /> [1roiff6y](https://wandb.ai/dl-project2/universal-computation-engine/runs/1roiff6y) <br /> [rdqaxnlm](https://wandb.ai/dl-project2/universal-computation-engine/runs/rdqaxnlm)
CIFAR 10 | Accuracy | 56.06% <br /> 58.54% <br /> 58.08% | 3 | 375 <br /> 550 <br /> 520 | steps_per_iter=200 <br /> test_steps_per_iter=50  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=4 | [2i50a27d](https://wandb.ai/dl-project2/universal-computation-engine/runs/2i50a27d) <br /> [1t4rqtyu](https://wandb.ai/dl-project2/universal-computation-engine/runs/1t4rqtyu) <br /> [2h7i4yza](https://wandb.ai/dl-project2/universal-computation-engine/runs/2h7i4yza)
MNIST Digits Addition | MSE | 7.543 <br /> 7.501 <br /> 7.69 | 3 | 150 <br /> 200 <br /> 90 | steps_per_iter=200 <br /> test_steps_per_iter=50  <br /> learning_rate=1e-3  <br /> batch_size=16 <br /> patch_size=28 <br /> n=10 | [1dj9wo3f](https://wandb.ai/dl-project2/universal-computation-engine/runs/1dj9wo3f) <br /> [1drqzu9o](https://wandb.ai/dl-project2/universal-computation-engine/runs/1drqzu9o) <br /> [1x5p2yog](https://wandb.ai/dl-project2/universal-computation-engine/runs/1x5p2yog)
Cyp3A4 Inhibition | Accuracy | 73.08% <br /> 75.8% <br /> 76.52% | 3 | 50 <br /> 75 <br /> 280 |steps_per_iter=200 <br /> test_steps_per_iter=50  <br /> learning_rate=1e-3  <br /> batch_size=16 | [fitarqv4](https://wandb.ai/dl-project2/universal-computation-engine/runs/fitarqv4) <br /> [3cnuxa09](https://wandb.ai/dl-project2/universal-computation-engine/runs/3cnuxa09) <br /> [3d155atc](https://wandb.ai/dl-project2/universal-computation-engine/runs/3d155atc)
Speech Command | Accuracy | 32.87% <br /> 24.94% <br /> 37.44% | 3 | 240 <br /> 120 <br /> 350 | steps_per_iter=200 <br /> test_steps_per_iter=25  <br /> learning_rate=1e-4  <br /> batch_size=16 <br /> patch_size=80| [1abgqjcm](https://wandb.ai/dl-project2/universal-computation-engine/runs/1abgqjcm) <br /> [2e3sow6q](https://wandb.ai/dl-project2/universal-computation-engine/runs/2e3sow6q) <br /> [voonklgk](https://wandb.ai/dl-project2/universal-computation-engine/runs/voonklgk)

### Conclusions

* Comparison with baselines:

Model | MNIST | CIFAR10 | MNIST Digits Addition | Cyp3A4 Inhibition | Speech Command
:---: | :---: | :---: | :---: | :---: | :---:
FRT (mean) | 97.05% | 57.56% | 7.578 | 75.13% | 31.75%
FRT (best) | 97.32% | 58.54% | 7.501 | 76.52% | 37.44%
UPT | 60.76% | 21.73% | 7.41* | 61.82% | 63.56%
FPT | 98.15% | 63.24% | 7.404* | 75.65% | 64.97%
Baseline | 99.5% | 99.5% | 1.42 | 82.1% | 98.1%
## Question 4

Can pretrained visual models transfer to different modalities?

### Methodology

0. Implement using ViT as the pretrained transformer.
1. Train Visual Frozen Pretrained Transformer (V-FPT) on all datasets with default parameters set, but with ViT as pretrained transformer:

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

2. Compare the results with the results of [baselines](#Baselines) and the results from [question 1](#Question 1). Are
   they somehow comparable?

### Empirical results

Dataset | Metric Name | # runs | mean | std | weights | Reference
:---: | :---: | :---: | :---: | :---: | :---: | :---:
MNIST | Accuracy | 3 | 73.59% | 0.0626 | 59,146 | [WandB]()
CIFAR 10 | Accuracy | 1 | 44.58% | NaN | 83,722 | [WandB]()
MNIST Digits Addition (Regression, n=10) | Mean Absolute Error | 1 | 7.29 | NaN | 642,049 | [WandB](https://wandb.ai/dl-project2/universal-computation-engine/groups/mnist-add-reg-unfreeze-mnist-add-reg)
MNIST Digits Addition (Classification, n=10) | Mean Absolute Error | 2 | 1.79 | 0.3288 | 711,259 | [WandB]()
Cyp3A4 Inhibition | Accuracy | 3 | 73.87% | 0.0204 | 807,938 | [WandB]()
Speech Command | Accuracy | 1 | 36.16% | NaN | 127,523 | [WandB]()


### Conclusions

- ViT also perform quite well as Universal Computation Engine

Model | MNIST | CIFAR10 | MNIST Digits Addition | Cyp3A4 Inhibition | Speech Command
:---: | :---: | :---: | :---: | :---: | :---:
FRT | 97.08% | 58.54% | 7.501* | 76.52% | 37.44%
UPT | 60.76% | 21.73% | 7.41* | 61.82% | 63.56%
FPT | 98.15% | 63.24% | 7.404* | 75.65% | 64.97%
V-FPT | 73.59% | 44.58% | 7.29* | 73.87% | 36.16%
Baseline | 99.5% | 99.5% | 1.42 | 82.1% | 98.1%

## Experiment 1

Does pretraining scenario influence FPT accuracy?

### Methodology

1. Multiple pretrained transformers (gpt2 based) has been selected from HuggingFaces with:
  - different pretraining languges: `uer/gpt2-chinese-poem`, `LorenzoDeMattei/GePpeTto`, `rinna/japanese-gpt2-medium`, ...
  - different model size (embeddings size, number of attention heads): `tiny-gpt2`, `gpt2`, `gpt2-medium`, `gpt2-large`, ...
  - different kinds of specialities: `magic-the-gathering`, `gpt2-chess-uci`, `CodeGPT-small-py`, ...

2. Selected models has been pretrained and tested on various tasks

### Empirical results

> For full results take a look at `experiments/Results.ipynb` notebook

#### Task: MNIST 

pretrained model | mean accuracy | # trainable weights
:---: | :---: | :---:
ceostroff/harry-potter-gpt2-fanfiction | 96.28% | 59,146
*gpt2* | 95.94% | 59,146
sberbank-ai/rugpt3small_based_on_gpt2 |95.37% | 59,146
... | ... | ...
shtoshni/gpt2-chess-uci | 90.11% | 59,146
minimaxir/magic-the-gathering | 69.79% | 7,818
sshleifer/tiny-gpt2 | 14.74% | 84

#### Task: MNIST Digits Addition (N=10)

pretrained model | mean MAE | # trainable weights
:---: | :---: | :---:
ceostroff/harry-potter-gpt2-fanfiction | 1.7950 | 14,97,691
chrisliu298/arxiv_ai_gpt2 | 1.9036 | 1,308,251
distilgpt2 | 1.9068 | 1,479,259
gpt2 | 2.0455 | 1,497,691
... | ... | ...
minimaxir/magic-the-gathering |	3.2861 | 149,339
shtoshni/gpt2-chess-uci | 3.4264 | 1,104,475
sshleifer/tiny-gpt2 | 7.4578 | 3,911


#### Task: Bit-XOR (N=10)

pretrained model | mean accuracy | # trainable weights
:---: | :---: | :---:
gpt2 | 72.51% | 62,228
sberbank-ai/rugpt3small_based_on_gpt2 | 66.70% | 62,228
microsoft/CodeGPT-small-py | 66.70% | 62,228
... | ... | ...
gpt2-large | 49.93% | 226,580
microsoft/DialoGPT-medium | 49.79% | 132116
chrisliu298/arxiv_ai_gpt2 | 49.76% | 226,580


### Conclusion

- Number of trained weights is correlated with overall model score
- Additional pretraining on special domain may additionaly increase model performance
