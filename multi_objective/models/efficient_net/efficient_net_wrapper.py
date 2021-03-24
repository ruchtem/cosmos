from models.efficient_net.model import EfficientNet
import torch

# small wrapper
class EfficientNetWrapper(EfficientNet):


    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

        # self.private_00 = torch.nn.Linear(1000, 1)
        # self.private_01 = torch.nn.Linear(1000, 1)
        # self.private_02 = torch.nn.Linear(1000, 1)
        # self.private_03 = torch.nn.Linear(1000, 1)
        # self.private_04 = torch.nn.Linear(1000, 1)
        # self.private_05 = torch.nn.Linear(1000, 1)
        # self.private_06 = torch.nn.Linear(1000, 1)
        # self.private_07 = torch.nn.Linear(1000, 1)
        # self.private_08 = torch.nn.Linear(1000, 1)
        # self.private_09 = torch.nn.Linear(1000, 1)
        # self.private_10 = torch.nn.Linear(1000, 1)
        # self.private_11 = torch.nn.Linear(1000, 1)
        # self.private_12 = torch.nn.Linear(1000, 1)
        # self.private_13 = torch.nn.Linear(1000, 1)
        # self.private_14 = torch.nn.Linear(1000, 1)
        # self.private_15 = torch.nn.Linear(1000, 1)
        # self.private_16 = torch.nn.Linear(1000, 1)
        # self.private_17 = torch.nn.Linear(1000, 1)
        # self.private_18 = torch.nn.Linear(1000, 1)
        # self.private_19 = torch.nn.Linear(1000, 1)
        # self.private_20 = torch.nn.Linear(1000, 1)
        # self.private_21 = torch.nn.Linear(1000, 1)
        # self.private_22 = torch.nn.Linear(1000, 1)
        # self.private_23 = torch.nn.Linear(1000, 1)
        # self.private_24 = torch.nn.Linear(1000, 1)
        # self.private_25 = torch.nn.Linear(1000, 1)
        # self.private_26 = torch.nn.Linear(1000, 1)
        # self.private_27 = torch.nn.Linear(1000, 1)
        # self.private_28 = torch.nn.Linear(1000, 1)
        # self.private_29 = torch.nn.Linear(1000, 1)
        # self.private_30 = torch.nn.Linear(1000, 1)
        # self.private_31 = torch.nn.Linear(1000, 1)
        # self.private_32 = torch.nn.Linear(1000, 1)
        # self.private_33 = torch.nn.Linear(1000, 1)
        # self.private_34 = torch.nn.Linear(1000, 1)
        # self.private_35 = torch.nn.Linear(1000, 1)
        # self.private_36 = torch.nn.Linear(1000, 1)
        # self.private_37 = torch.nn.Linear(1000, 1)
        # self.private_38 = torch.nn.Linear(1000, 1)
        # self.private_39 = torch.nn.Linear(1000, 1)


    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
    
        
    def forward_linear(self, x, i):
        # x = self._dropout(x)
        # x = self._fc(x)
        result = {f'logits_{i}': self.task_layers[f'task_fc_{i}'](x)}
        return result
    
    def private_params(self):
        return [n for n, p in self.named_parameters() if "task_layers" not in n]


    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result
        # result = dict(
        #     logits_0=self.private_00(x),
        #     logits_1=self.private_01(x),
        #     logits_2=self.private_02(x),
        #     logits_3=self.private_03(x),
        #     logits_4=self.private_04(x),
        #     logits_5=self.private_05(x),
        #     logits_6=self.private_06(x),
        #     logits_7=self.private_07(x),
        #     logits_8=self.private_08(x),
        #     logits_9=self.private_09(x),
        #     logits_10=self.private_10(x),
        #     logits_11=self.private_11(x),
        #     logits_12=self.private_12(x),
        #     logits_13=self.private_13(x),
        #     logits_14=self.private_14(x),
        #     logits_15=self.private_15(x),
        #     logits_16=self.private_16(x),
        #     logits_17=self.private_17(x),
        #     logits_18=self.private_18(x),
        #     logits_19=self.private_19(x),
        #     logits_20=self.private_20(x),
        #     logits_21=self.private_21(x),
        #     logits_22=self.private_22(x),
        #     logits_23=self.private_23(x),
        #     logits_24=self.private_24(x),
        #     logits_25=self.private_25(x),
        #     logits_26=self.private_26(x),
        #     logits_27=self.private_27(x),
        #     logits_28=self.private_28(x),
        #     logits_29=self.private_29(x),
        #     logits_30=self.private_30(x),
        #     logits_31=self.private_31(x),
        #     logits_32=self.private_32(x),
        #     logits_33=self.private_33(x),
        #     logits_34=self.private_34(x),
        #     logits_35=self.private_35(x),
        #     logits_36=self.private_36(x),
        #     logits_37=self.private_37(x),
        #     logits_38=self.private_38(x),
        #     logits_39=self.private_39(x),
        # )
        # b = x.shape[0]
        # if self.late_fusion:
        #     a = batch['alpha'].repeat(b, 1)
        #     features = self.forward_feature_extraction(batch)
        #     x = torch.cat((features, a), dim=1)
        # else:
        #     x = super().forward(x)
        # result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result

    
    @classmethod
    def from_pretrained(cls, dim, task_ids, model_name, **override_params):
        cls.task_ids = task_ids
        cls.my_in_channels=dim[0]
        return super().from_pretrained(model_name, in_channels=dim[0], num_classes=len(task_ids))
    

    # @classmethod
    # def from_name(cls, model_name, **override_params):

    #     return super().from_name(model_name, **override_params)


