# modified https://www.kaggle.com/code/skraiii/pppm-tokenclassificationmodel-train-8th-place

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

class AWP:
    def __init__(
        self,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        apex: bool,
        adv_param: str="weight",
        adv_lr: float=1.0,
        adv_eps: float=0.01
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.apex = apex
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs: dict) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.apex):
            self._save()
            self._attack_step() # モデルを近傍の悪い方へ改変
            #y_preds = self.model(inputs)
            _, _, adv_loss = self.model.training_step(inputs)
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 直前に損失関数に通してパラメータの勾配を取得できるようにしておく必要あり
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}