# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset

class CamVid(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_samples=None,
                 num_classes=11,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=960, 
                 crop_size=(512, 512),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(CamVid, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std, )

        project_path = os.path.abspath(os.path.join(os.getcwd(), '..','..'))
        self.root = os.path.join(project_path, root)

        self.img_path = list_path

        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(self.root+'/'+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.ignore_label = ignore_label
        
        self.color_list = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                             [192, 192, 128], [64, 64, 128], [64, 64, 0],
                             [128, 64, 128], [0, 0, 192], [192, 128, 128],
                             [128, 128, 128], [128, 128, 0]]
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
            
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]
            
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(os.path.join(self.root,item["img"])).convert('RGB')
        image = np.array(image)
        size = image.shape

        color_map = Image.open(os.path.join(self.root,item["label"])).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy()

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        