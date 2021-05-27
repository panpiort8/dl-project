import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
import wandb
import json

from universal_computation.experiment import run_experiment
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--tag', action='append')
    parser.add_argument('--task-name', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('config')
    
    args = parser.parse_args()
    print(args)
    
    ####### Load configuration
    with open(args.config) as input:
        params = json.load(input)

    experiment_name = params['name']

    experiment_params = params['params']

    if args.task_name:
        *prefix, suffix = args.task_name.split('-')
        if suffix.isdigit():
            experiment_params['task'] = '-'.join(prefix)
            experiment_params['n'] = int(suffix)
        else:
            experiment_params['task'] = args.task_name
   
    if args.model:
        experiment_params['model_name'] = args.model

    from types import SimpleNamespace

    Args = SimpleNamespace(**params['args'])
    
    ####### Run experiment
    trainer = run_experiment(experiment_name, experiment_params, Args)

    final = trainer.test_eval(10000, None)

    wandb.log({
        'Final Loss': final[0],
        'Final Accuracy': final[1],
        'Final Test Size': final[2]
    })

    ####### Finalize
    wandb.run.tags += ("completed",)
    if args.tag:
        wandb.run.tags += tuple(args.tag)

    wandb.run.finish()