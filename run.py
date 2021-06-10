#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
import wandb
import json

from universal_computation.experiment import run_experiment
from argparse import ArgumentParser

DEFAULT_PARAMS = {
    "model_name": "gpt2",
    "pretrained": True,

    "freeze_trans": True,
    "freeze_in": False,
    "freeze_pos": True,
    "freeze_ln": False,
    "freeze_attn": True,
    "freeze_ff": True,
    "freeze_out": False,

    "in_layer_sizes": None,
    "out_layer_sizes": None,

    "learning_rate": 0.001,
    "batch_size": 25,
    "dropout": 0.1,
    "orth_gain": 1.41,
    "patch_size": None
}

DEFAULT_ARGS = {
    "log_to_wandb": True,
    "note": "",
    "wandb_project": "universal-computation-engine",
    "wandb_entity": "dl-project2",

    "include_date": False,
    "save_models": False,
    "save_models_ever": 25,
    "device": "cuda",

    "num_iters": 50,
    "steps_per_iter": 1000,
    "test_steps_per_iter": 100,
    "gpu_batch_size": 25    
}


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--tag', action='append')
    parser.add_argument('--task-name', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--early-stop', type=int)
    parser.add_argument('config')
    
    args = parser.parse_args()
    print(args)
    

    ####### Load configuration
    with open(args.config) as input:
        params = json.load(input)

    experiment_name = params['name']

    experiment_params = {
        **DEFAULT_PARAMS,
        **params['params']
    }

    if args.task_name:
        *prefix, suffix = args.task_name.split('-')
        if suffix.isdigit():
            experiment_params['task'] = '-'.join(prefix)
            experiment_params['n'] = int(suffix)
        else:
            experiment_params['task'] = args.task_name
   
    if args.model:
        experiment_params['model_name'] = args.model
       
    if args.early_stop:
        experiment_params['early_stop'] = int(args.early_stop)

    from types import SimpleNamespace

    experiment_args = {
        **DEFAULT_ARGS,
        **params['args']
    }
    Args = SimpleNamespace(**experiment_args)
    

    ####### Run experiment
    trainer = run_experiment(experiment_name, experiment_params, Args)

    final = trainer.test_eval(10000, experiment_params['batch_size'])

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
