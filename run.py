import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
import wandb

from universal_computation.experiment import run_experiment

#######

import json
with open(sys.argv[1]) as input:
    params = json.load(input)
    

experiment_name = params['name']

experiment_params = params['params']

from types import SimpleNamespace

Args = SimpleNamespace(**params['args'])

#######

trainer = run_experiment(experiment_name, experiment_params, Args)

final = trainer.test_eval(50, None)

wandb.log({
    'Final Loss': final[0],
    'Final Accuracy': final[1],
    'Final Test Size': final[2]
})

#######


wandb.run.tags += ("completed",)
# wandb.run.update()

wandb.run.finish()