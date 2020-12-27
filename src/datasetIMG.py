import torch
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np
from xml.dom import minidom 
import cv2
import math 

class DataLoaderInstanceSegmentation(Dataset):
    def __init__(self, folder_path="inrae_1_all", train = True):
        super(DataLoaderInstanceSegmentation, self).__init__()
        if train:
            folder_path="inrae_1_all"
        else:     
            folder_path="images_testing"
        self.train = train
        self.img_files = glob.glob(os.path.join(folder_path,"backless_images","*.jpg"))
        self.seg_mask_files = []
        self.ins_mask_files = []
        self.to_tensor = transforms.ToTensor()
        for img_path in self.img_files:
            self.seg_mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path)))
            # self.ins_mask_files.append(os.path.join(folder_path,'insmasks',os.path.splitext(os.path.basename(img_path))[0]+'.xml'))
            self.ins_mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path)))


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

        data =  np.asarray(Image.open(img_path).convert('RGB')).transpose((2,0,1))
        data = torch.Tensor(data)
        # data =  np.asarray(Image.open(img_path).convert('L'))
        # data = torch.Tensor(data[np.newaxis])
        
        # data =  torch.Tensor(Image.open(img_path).convert('RGB'))
        # data =  self.to_tensor(Image.open(img_path).convert('RGB'))


        # data_shape = np.ones((1024, 1024), dtype=np.uint8) * 255



        # label_ins =  self.to_tensor(Image.open(ins_mask_path).convert('L'))
        # label_ins =  torch.Tensor(Image.open(ins_mask_path).convert('L'))

        label_ins =  np.asarray(Image.open(ins_mask_path).convert('L'))
        label_ins = torch.Tensor(label_ins[np.newaxis])

        # fullname = os.path.join(ins_mask_path)
        # xmldoc = minidom.parse(fullname)
        # itemlist = xmldoc.getElementsByTagName('robndbox')
        # ins = np.zeros((0, 1024, 1024), dtype=np.uint8)
        # for rec in itemlist[:40]:
        #     x = float(rec.getElementsByTagName('cx')[0].firstChild.nodeValue)
        #     y = float(rec.getElementsByTagName('cy')[0].firstChild.nodeValue)
        #     w = float(rec.getElementsByTagName('w')[0].firstChild.nodeValue) 
        #     h = float(rec.getElementsByTagName('h')[0].firstChild.nodeValue)
        #     theta = float(rec.getElementsByTagName('angle')[0].firstChild.nodeValue)
        #     rect = ([x, y], [w, h], math.degrees(theta))
        #     box = np.int0(cv2.boxPoints(rect))
        #     gt = np.zeros_like(data_shape)
        #     gt = cv2.fillPoly(gt, [box], 1)
        #     ins[:, gt != 0] = 0
        #     ins = np.concatenate([ins, gt[np.newaxis]])


        
        # sem = np.zeros_like(data_shape, dtype=bool)
        # sem[np.sum(ins, axis=0) != 0] = True
        # sem = np.stack([~sem, sem]).astype(np.uint8)


        # label_ins = torch.Tensor(ins)
        # label_seg = torch.Tensor(sem)

        
        # label_seg =  torch.Tensor(Image.open(seg_mask_path).convert('L'))
        # label_seg =  self.to_tensor(Image.open(seg_mask_path).convert('L'))

        label_seg =  np.asarray(Image.open(seg_mask_path).convert('L'))
        label_seg = torch.Tensor(label_seg[np.newaxis])

        if self.train:
            return data, label_seg, label_ins
        else:     
            return data