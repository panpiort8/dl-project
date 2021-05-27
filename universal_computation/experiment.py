import random
from datetime import datetime

import numpy as np
import torch
import wandb

from universal_computation.fpt import FPT
from universal_computation.trainer import Trainer

def count_weights(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def experiment(
        exp_name,
        exp_args,
        early_stop: int = 10,
        **kwargs
):
    """
    Preliminary checks
    """

    # Must be able to accumulate gradient if batch size is large
    assert 'batch_size' in kwargs
    assert kwargs['batch_size'] <= exp_args['gpu_batch_size'] or \
           kwargs['batch_size'] % exp_args['gpu_batch_size'] == 0

    """
    Create dataset, model, and trainer
    """

    task = kwargs['task']
    batch_size = kwargs['batch_size']
    patch_size = kwargs['patch_size']
    device = exp_args['device']

    return_last_only = True

    if task == 'bit-memory':
        from universal_computation.datasets.bit_memory import BitMemoryDataset
        dataset = BitMemoryDataset(n=kwargs['n'], num_patterns=kwargs['num_patterns'], device=device)
        input_dim = kwargs['n'] if patch_size is None else patch_size
        output_dim = 2 * kwargs['n'] if patch_size is None else 2 * patch_size
        use_embeddings = False
        experiment_type = 'classification'

    elif task == 'bit-xor':
        from universal_computation.datasets.bit_xor import BitXORDataset
        dataset = BitXORDataset(n=kwargs['n'], num_patterns=kwargs['num_patterns'], device=device)
        input_dim = kwargs['n'] if patch_size is None else patch_size
        output_dim = 2 * kwargs['n'] if patch_size is None else 2 * patch_size
        use_embeddings = False
        experiment_type = 'classification'

    elif task == 'mnist':
        from universal_computation.datasets.mnist import MNISTDataset
        dataset = MNISTDataset(batch_size=batch_size, patch_size=patch_size, device=device)
        input_dim, output_dim = patch_size ** 2, 10
        use_embeddings = False
        experiment_type = 'classification'

    elif task == 'mnist-add':
        from universal_computation.datasets.mnist import MNISTDigitAdditionDataset
        dataset = MNISTDigitAdditionDataset(batch_size=batch_size, seq_length=kwargs['n'], device=device)
        input_dim, output_dim = 28**2, 9*kwargs['n']+1
        use_embeddings = False
        experiment_type = 'classification'
        
    elif task == 'mnist-add-reg':
        from universal_computation.datasets.mnist import MNISTDigitAdditionDataset
        dataset = MNISTDigitAdditionDataset(batch_size=batch_size, seq_length=kwargs['n'], device=device)
        input_dim, output_dim = 28**2, 1
        use_embeddings = False
        experiment_type = 'regression'
        
    elif task == 'cifar10':
        from universal_computation.datasets.cifar10 import CIFAR10Dataset
        dataset = CIFAR10Dataset(batch_size=batch_size, patch_size=patch_size, device=device)
        input_dim, output_dim = 3 * patch_size ** 2, 10
        use_embeddings = False
        experiment_type = 'classification'

    elif task == 'cifar10-gray':
        from universal_computation.datasets.cifar10_gray import CIFAR10GrayDataset
        dataset = CIFAR10GrayDataset(batch_size=batch_size, patch_size=patch_size, device=device)
        input_dim, output_dim = patch_size ** 2, 10
        use_embeddings = False
        experiment_type = 'classification'

    elif task == 'listops':
        from universal_computation.datasets.listops import ListopsDataset
        dataset = ListopsDataset(batch_size=batch_size, device=device)
        input_dim, output_dim = 15, 10
        use_embeddings = True
        experiment_type = 'classification'

    elif task == 'cyp3a4':
        from universal_computation.datasets.cyp3a4 import Cyp3A4Dataset
        dataset = Cyp3A4Dataset(batch_size=batch_size, device=device)
        input_dim, output_dim = 1000, 2
        use_embeddings = True
        experiment_type = 'classification'

    else:
        raise NotImplementedError('dataset not implemented')

    if 'bit' in task:

        ce_loss = torch.nn.CrossEntropyLoss()

        def loss_fn(out, y, x=None):
            out = torch.reshape(out, (-1, kwargs['n'], 2))
            ids = torch.zeros(y.shape).to(device=y.device).long()
            if task == 'bit-memory':
                ids[y < 0], ids[y > 0] = 0, 1
            else:
                ids[y < 0.5], ids[y > 0.5] = 0, 1
            out, ids = torch.reshape(out, (-1, 2)), torch.reshape(ids, (-1,))
            return ce_loss(out, ids)

        def accuracy_fn(preds, true, x=None):
            if task == 'bit-memory':
                preds = preds.reshape(-1, kwargs['n'], 2).argmax(-1) * 2 - 1
            else:
                preds = preds.reshape(-1, kwargs['n'], 2).argmax(-1)
            if task == 'bit-memory':
                return (np.sign(preds) == np.sign(true)).mean()
            else:
                return ((preds > 0.5) == (true > 0.5)).mean()
    elif task == 'mnist-add':
        ce_loss = torch.nn.CrossEntropyLoss()

        def loss_fn(out, y, x=None):
            out = out[:, 0]
            return ce_loss(out, y)

        mae_loss = torch.nn.L1Loss()
        def accuracy_fn(preds, true, x=None):
            preds = preds[:, 0].argmax(-1)
            return np.abs(preds - true).mean()
#             print(preds, true)
#             return mae_loss(preds, true)
        
    elif experiment_type == 'regression':
        def loss_fn(out, y, x=None):
            out = out[:, 0]
            return torch.abs(out - y).mean()

        def accuracy_fn(preds, true, x=None):
            return np.abs(preds - true).mean()
        
    elif experiment_type == 'classification':

        ce_loss = torch.nn.CrossEntropyLoss()

        def loss_fn(out, y, x=None):
            out = out[:, 0]
            return ce_loss(out, y)

        def accuracy_fn(preds, true, x=None):
            preds = preds[:, 0].argmax(-1)
            return (preds == true).mean()

    else:
        raise NotImplementedError('experiment_type not recognized')

    model = FPT(
        input_dim=input_dim,
        output_dim=output_dim,
        model_name=kwargs.get('model_name', 'gpt2'),
        pretrained=kwargs.get('pretrained', True),
        return_last_only=return_last_only,
        use_embeddings_for_in=use_embeddings,
        in_layer_sizes=kwargs.get('in_layer_sizes', None),
        out_layer_sizes=kwargs.get('out_layer_sizes', None),
        freeze_trans=kwargs.get('freeze_trans', True),
        freeze_in=kwargs.get('freeze_in', False),
        freeze_pos=kwargs.get('freeze_pos', False),
        freeze_ln=kwargs.get('freeze_ln', False),
        freeze_attn=kwargs.get('freeze_attn', True),
        freeze_ff=kwargs.get('freeze_ff', True),
        freeze_out=kwargs.get('freeze_out', False),
        dropout=kwargs['dropout'],
        orth_gain=kwargs['orth_gain'],
    )
    model.to(device)

    gpu_batch_size = exp_args['gpu_batch_size']
    trainer = Trainer(
        model,
        dataset,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        steps_per_epoch=exp_args['steps_per_iter'],
        test_steps_per_epoch=exp_args['test_steps_per_iter'],
        learning_rate=kwargs['learning_rate'],
        batch_size=gpu_batch_size if batch_size > gpu_batch_size else batch_size,
        eval_batch_size=batch_size,
        grad_accumulate=batch_size // gpu_batch_size if batch_size > gpu_batch_size else 1,
    )

    """
    Set up logging
    """

    log_to_wandb = exp_args['log_to_wandb']
    save_models = exp_args['save_models']
    wandb_project = exp_args['wandb_project']

    short_name = datetime.now().strftime('%Y%m%d-%H%M')
#     short_name = str(random.randint(int(1e5), int(1e6) - 1))
    run_name = f'{exp_name}-{task}-{short_name}'

    if log_to_wandb:
        config = dict(
            short_name=short_name,
            run_name=run_name,
            **exp_args,
            **kwargs,
        )
        if '__dict__' in config:
            del config['__dict__']
        if '__weakref__' in config:
            del config['__weakref__']
        print(config)
        print('*'*100)
        wandb.init(
            name=f'{exp_name}-{short_name}',
            group=f'{exp_name}-{task}-n-{kwargs["n"]}',
            project='universal-computation-engine',
            entity='dl-project2',
            config=config,
        )
        print('*'*100)
        wandb.watch(model)

    best_test_loss = 1e10
    best_test_iter = -1
    print(model)
    wandb.log({"model_weights": count_weights(model)})
    for t in range(exp_args['num_iters']):
        trainer.train_epoch()

        print('=' * 57)
        print(f'| Iteration {" " * 15} | {t + 1:25} |')
        for k, v in trainer.diagnostics.items():
            print(f'| {k:25} | {v:25} |')

        if log_to_wandb:
            wandb.log(trainer.diagnostics)

        if best_test_loss > trainer.diagnostics['Test Loss']:
            best_test_loss = trainer.diagnostics['Test Loss']
            best_test_iter = t
            
            with open(f'models/{run_name}.pt', 'wb') as f:
                state_dict = dict(model=model.state_dict(), optim=trainer.optim.state_dict())
                torch.save(state_dict, f)
            
            print(f'Saved model at {t + 1} iters: {run_name}')
            
        if t - best_test_iter >= early_stop:
            print(f'No progress since {early_stop} epoch. Early stopping.')
            print('Loading best model!')
            state = torch.load(f'models/{run_name}.pt')
            model.load_state_dict(state['model'])
            trainer.optim.load_state_dict(state['optim'])
            
            break
    
    
    return trainer


def run_experiment(
        exp_name,
        experiment_params,
        experiment_args
):
    exp_args = experiment_args

    if exp_args.include_date:
        timestamp = datetime.now().strftime('%m-%d')
        exp_name = f'{timestamp}-{exp_name}'

    experiment_params['exp_name'] = exp_name
    experiment_params['exp_args'] = vars(exp_args)

    return experiment(xp_name=exp_name, **experiment_params)
