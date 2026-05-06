"""
PCGrad: Gradient Surgery for Multi-Task Learning
Matches https://github.com/WeiChengTseng/Pytorch-PCGrad
"""
import torch
import copy
import random


class PCGrad:
    def __init__(self, optimizer, reduction='mean'):
        self._optim = optimizer
        self._reduction = reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self._optim.param_groups

    def pc_backward(self, objectives):
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for i, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            # retain graph only until the last loss; they share the same forward graph
            obj.backward(retain_graph=(i < len(objectives) - 1))
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _project_conflicting(self, grads, has_grads):
        # params that have gradients from ALL tasks
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            # random ordering matches the original repo
            grads_copy = copy.deepcopy(grads)
            random.shuffle(grads_copy)
            for g_j in grads_copy:
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    g_i -= dot * g_j / (g_j.norm() ** 2)
        merged = torch.zeros_like(grads[0])
        if self._reduction == 'mean':
            merged[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        # non-shared params: accumulate each task's own gradient
        for g in pc_grad:
            merged[~shared] += g[~shared]
        return merged

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                shape.append(p.shape)
                if p.grad is None:
                    grad.append(torch.zeros_like(p))
                    has_grad.append(torch.zeros_like(p))
                else:
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p))
        return grad, shape, has_grad

    def _flatten_grad(self, grads, shapes):
        return torch.cat([g.view(-1) for g in grads])

    def _unflatten_grad(self, flat_grad, shapes):
        grads, idx = [], 0
        for shape in shapes:
            n = 1
            for s in shape:
                n *= s
            grads.append(flat_grad[idx: idx + n].view(shape))
            idx += n
        return grads

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1