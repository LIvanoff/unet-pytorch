import torch
import torch.nn as nn

from models import UNetTransposed, UNetMaxUnpool


class UNet(nn.Module):
    def __init__(self,
                 pooling: str,
                 activ: str = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if pooling is not None:
            if pooling == 'maxpooling':
                self.backbone = UNetMaxUnpool()
            elif pooling == 'transposed':
                self.backbone = UNetTransposed()
            else:
                raise NotImplementedError(f'Pooling {pooling} not implemented.\n'
                                          f'You can use follow activation functions: maxpooling or transposed')

        if activ is not None:
            self.is_activ = True
            if activ == 'sigmoid':
                self.activ = nn.Sigmoid()
            elif activ == 'softmax':
                self.activ = nn.Softmax(dim=1)
            else:
                raise NotImplementedError(f'Activation function {activ} not implemented.\n'
                                          f'You can use follow activation functions: sigmoid or softmax')

    def forward(self, x):
        out = self.backbone(x)
        if self.is_activ is not None:
            out = self.activ(out)
        return out
