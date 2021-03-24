import os
import torch
import numpy as np
import re
import glob
import torchvision.transforms as transforms
from PIL import Image

from torch.utils import data

# taken from https://github.com/intel-isl/MultiObjectiveOptimization and adapted


class CelebA(data.Dataset):
    def __init__(self, split, task_ids=[], root='data/celeba', dim=64, augmentations=None, **kwargs):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.task_ids = task_ids
        self.augmentations = augmentations
        self.n_classes =  40
        self.files = {}
        self.labels = {}

        assert dim[-1] == dim[-2]

        self.transform=transforms.Compose([
            transforms.Resize(dim[-1]),
            transforms.CenterCrop(dim[-1]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.label_file = self.root+"/Anno/list_attr_celeba.txt"
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[2:-1]
        for label_line in labels:
            f_name = re.sub('jpg', 'png', label_line.split(' ')[0])
            label_txt = list(map(lambda x:int(x), re.sub('-1','0',label_line).split()[1:]))
            label_map[f_name]=label_txt

        self.all_files = glob.glob(self.root+'/Img/img_align_celeba_png/*.png')
        with open(root+'//Eval/list_eval_partition.txt', 'r') as f:
            fl = f.read().split('\n')
            fl.pop()
            if 'train' in self.split:
                selected_files = list(filter(lambda x:x.split(' ')[1]=='0', fl))
            elif 'val' in self.split:
                selected_files =  list(filter(lambda x:x.split(' ')[1]=='1', fl))
            elif 'test' in self.split:
                selected_files =  list(filter(lambda x:x.split(' ')[1]=='2', fl))
            selected_file_names = list(map(lambda x:re.sub('jpg', 'png', x.split(' ')[0]), selected_files))
        
        base_path = '/'.join(self.all_files[0].split('/')[:-1])
        self.files[self.split] = list(map(lambda x: '/'.join([base_path, x]), set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.labels[self.split] = list(map(lambda x: label_map[x], set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                                'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',      
                                'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',       
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
                                'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
                                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        if len(self.files[self.split]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

        print("Found {} {} images. Defined tasks: {}".format(
            len(self.files[self.split]), 
            self.split,
            [self.class_names[i] for i in task_ids] if task_ids else 'all'
        ))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        label = self.labels[self.split][index]
        label = torch.Tensor(label).long()

        img_path = self.files[self.split][index].rstrip()
        img = Image.open(img_path)

        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))

        img = self.transform(img)
        labels = {'labels_{}'.format(i): label[i] for i in self.task_names()}
        return dict(data=img, **labels)

    
    def task_names(self):
        return self.task_ids if self.task_ids else range(self.n_classes)


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    dst = CelebA(split='val', task_ids=[22, 39])
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    for i, data in enumerate(trainloader):
        imgs = data['data']
        labels = data['labels']
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])

        f, axarr = plt.subplots(bs,4)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()
