# dl-project

## Overview

Reproduces and extends [Pretrained Transformers as Universal Computation Engines](https://arxiv.org/abs/2103.05247).

### Pipeline

Open our pipeline in [Colab](https://colab.research.google.com/github/panpiort8/dl-project/blob/master/pipeline.ipynb).

### Datasets

#### Cyp3A4 Inhibition

The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within
cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It
oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the
body.

### Baselines

Dataset | Accuracy
:---: | :---:
Cyp3A4 Inhibition | ...

#### Cyp3A4 Inhibition

The baseline for this dataset is Molecule Attention Transformer, fine-tuned with [huggingmolecules](https://github.com/gmum/huggingmolecules) package (on the same data split, with a default hps setting).