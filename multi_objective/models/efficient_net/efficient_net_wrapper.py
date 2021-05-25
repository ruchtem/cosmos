from models.efficient_net.model import EfficientNet
import torch

# small wrapper
class EfficientNetWrapper(EfficientNet):


    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

        self.task_layers = torch.nn.ModuleDict({
            f'task_fc_{t}': torch.nn.Linear(1792, 1) for t in self.task_ids
        })


    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
    
        
    def forward_linear(self, x, i):
        result = {f'logits_{i}': self.task_layers[f'task_fc_{i}'](x)}
        return result
    
    def private_params(self):
        return [n for n, p in self.named_parameters() if "task_layers" not in n]


    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result

    
    @classmethod
    def from_pretrained(cls, dim, task_ids, model_name, **override_params):
        cls.task_ids = task_ids
        cls.my_in_channels=dim[0]
        return super().from_pretrained(model_name, in_channels=dim[0], num_classes=len(task_ids))
