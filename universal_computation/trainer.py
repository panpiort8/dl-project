import torch
from tqdm import tqdm

import time


class Trainer:

    def __init__(
            self,
            model,
            dataset,
            loss_fn,
            accuracy_fn=None,
            steps_per_epoch=100,
            test_steps_per_epoch=20,
            learning_rate=1e-3,
            batch_size=2,
            eval_batch_size=8,
            grad_accumulate=1,
    ):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.acc_fn = accuracy_fn
        self.steps_per_epoch = steps_per_epoch
        self.test_steps_per_epoch = test_steps_per_epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.grad_accumulate = grad_accumulate

        self.optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.diagnostics = {'Gradient Steps': 0}

    def get_loss(self, x, y, return_acc=False):
        out = self.model(x)
        loss = self.loss_fn(out, y, x=x)
        if return_acc:
            if self.acc_fn is None:
                raise NotImplementedError('accuracy function not specified')
            accs = self.acc_fn(
                out.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                x=x.detach().cpu().numpy(),
            )
            return loss, accs
        return loss

    def test_eval(self, steps, batch_size):
        test_loss, accuracy, total = 0., 0., 0
        accs, losss =[], []
        with torch.no_grad():
            while total < steps:
                x, y = self.dataset.get_batch(batch_size, train=False)
                total += y.shape[0]
                loss, acc = self.get_loss(x, y, return_acc=True)
                losss.append(loss.detach().cpu().item())
                accs.append(acc)
#                 test_loss += loss.detach().cpu().item() / steps
#                 accuracy += acc * y.shape[0]
        return sum(losss)/len(losss), sum(accs)/len(accs), total
        return test_loss, accuracy, total
                
    def train_epoch(self, test_steps=None):
        self.dataset.start_epoch()

        train_losses, tr_accuracy = [], 0.
        self.model.train()
        start_train_time = time.time()
        self.sample_counts = 0
        for _ in tqdm(range(self.steps_per_epoch)):
            step_loss = 0
            for _ in range(self.grad_accumulate):
                x, y = self.dataset.get_batch(self.batch_size, train=True)
                self.sample_counts += x.shape[0]
                loss, acc = self.get_loss(x, y, return_acc=True)
                loss = loss / self.grad_accumulate
                loss.backward()
                step_loss += loss.detach().cpu().item()
                tr_accuracy += acc

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optim.step()
            self.optim.zero_grad()

            self.diagnostics['Gradient Steps'] += 1

            train_losses.append(step_loss)
        end_train_time = time.time()

        test_steps = self.test_steps_per_epoch if test_steps is None else test_steps

        
        self.model.eval()
        start_test_time = time.time()
        test_loss, accuracy, _ = self.test_eval(test_steps, self.eval_batch_size)
        end_test_time = time.time()

        self.diagnostics['Average Train Loss'] = sum(train_losses) / self.steps_per_epoch
        self.diagnostics['Start Train Loss'] = train_losses[0]
        self.diagnostics['Final Train Loss'] = train_losses[-1]
        self.diagnostics['Test Loss'] = test_loss
        self.diagnostics['Test Accuracy'] = accuracy
        self.diagnostics['Train Accuracy'] = tr_accuracy / (self.steps_per_epoch * self.grad_accumulate)
        self.diagnostics['Time Training'] = end_train_time - start_train_time
        self.diagnostics['Time Testing'] = end_test_time - start_test_time
