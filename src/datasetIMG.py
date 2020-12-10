import torch
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np 

class DataLoaderInstanceSegmentation(Dataset):
    def __init__(self, folder_path="ethz_1/images"):
        super(DataLoaderInstanceSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,"raw","*.jpg"))
        self.seg_mask_files = []
        self.ins_mask_files = []
        self.to_tensor = transforms.ToTensor()
        for img_path in self.img_files:
            self.seg_mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path)))
            self.ins_mask_files.append(os.path.join(folder_path,'insmasks',os.path.basename(img_path)))


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        seg_mask_path = self.seg_mask_files[index]
        ins_mask_path = self.ins_mask_files[index]
        # data = np.array(Image.open(img_path))
        # label_seg = np.array(Image.open(seg_mask_path))
        # label_ins = np.array(Image.open(ins_mask_path))
        # return torch.from_numpy(data).float(), torch.from_numpy(label_seg).float(), torch.from_numpy(label_ins).float()
        data =  self.to_tensor(Image.open(img_path))
        label_seg =  self.to_tensor(Image.open(seg_mask_path))
        label_ins =  self.to_tensor(Image.open(ins_mask_path))
        return data, label_seg, label_ins
