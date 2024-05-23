import torch
from torch.optim.optimizer import Optimizer, required
from estimator import ZOGradientEstimator
import numpy as np


class MeZO_SVRG(Optimizer):
    '''ban the backward propagation of the gradient'''

    def __init__(self, params, Gradient_Estimator, lr1=required, lr2=required, q=required, mu=0.01, batch_size=32,
                 full_batch_size=None):
        self.estimator = Gradient_Estimator
        self.q = q
        if not 0.0 <= lr1:
            raise ValueError("Invalid learning rate: {}".format(lr1))
        if not 0.0 <= lr2:
            raise ValueError("Invalid learning rate: {}".format(lr2))
        if not 0.0 <= q:
            raise ValueError("Invalid learning rate: {}".format(q))
        if not 0.0 <= batch_size:
            raise ValueError("Invalid learning rate: {}".format(batch_size))
        if not isinstance(Gradient_Estimator, ZOGradientEstimator):
            raise ValueError("Estimator type error")
        defaults = dict(lr1=lr1, lr2=lr2, mu=mu, batch_size=batch_size, full_batch_size=full_batch_size, q=q)
        super(MeZO_SVRG, self).__init__(params, defaults)

        self.state['step'] = 0

    def step(self, batch, closure=None):  # 所有参数都更新一遍
        loss = None
        if self.state['step'] % self.q == 0:
            self.estimator.compute_zeroth_order_grad(batch=batch, batch_mode=False, epsilon=1e-4)
            self.estimator.save_params()
        else:
            self.estimator.compute_zeroth_order_grad(batch=batch, current=True, batch_mode=True, epsilon=1e-4)
            self.estimator.copy_params()
            self.estimator.restore()
            self.estimator.compute_zeroth_order_grad(batch=batch, current=False, batch_mode=True, epsilon=1e-4)
            self.estimator.reload_params()
            '''
            构造另一套函数计算上一个参数值上的梯度估计
            '''
        for group in self.param_groups:
            for p in group['params']:
                lr1 = group['lr1']
                lr2 = group['lr2']
                q = group['q']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # previous full batch gradient
                    state['full_grad_pre'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                full_grad = state['full_grad_pre']

                if state['step'] % q == 0:
                    full_grad.copy_(p.grad.data)
                    p.data.add_(-lr1, p.grad.data)
                    full_grad.copy_(p.grad.data)
                else:
                    p.data.add_(-lr2, p.grad.data)
                    p.data.add_(lr2, p.pre_grad.data)
                    p.data.add_(-lr2, full_grad)

                state['step'] += 1

            self.state['step'] += 1

        return loss


# Example usage
def example_loss(theta):
    # Define a simple loss function for demonstration purposes
    return sum([torch.sum(t ** 2) for t in theta])


# Define a simple model for demonstration purposes
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc1(x)


