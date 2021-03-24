import torch
from .resnet import ResNet, BasicBlock, Bottleneck, model_urls
from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


class ResNetWrapper(ResNet):

    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
        
    def forward_linear(self, x):
        x = self.fc(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result

    
    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result


    @classmethod
    def from_name(cls, model_name, dim, task_ids, **override_params):
        cls.task_ids = task_ids
        return resnet18(
            pretrained=True, 
            progress=False,
            in_channels=dim[0],
            num_classes=1000,
            # num_classes=len(task_ids),
        )


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetWrapper:
    model = ResNetWrapper(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        
        # fix input dim
        state_dict['conv1.weight'] = torch.nn.parameter.Parameter(torch.ones((64, 5, 7, 7)))
        
        # remove bn params
        state_dict = {k:v for k, v in state_dict.items() if 'bn' not in k and 'downsample.1' not in k}
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, norm_layer=torch.nn.Identity,
                   **kwargs)