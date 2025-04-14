from model_architecture import device
import os
import csv
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


test_dir= os.path.join('dataset','test')
train_dir = os.path.join('dataset','train')
data_dirs = [test_dir,train_dir]
file_names = ['test','train']
header = ['file','age', 'gender']
train_data,test_data=[],[]
for idx,data_dir in enumerate(data_dirs):
    for img_name in os.listdir(data_dir):
        age  = img_name.split('_')[0]
        gender = img_name.split('_')[1] 
        if idx==1:
            path_img = 'train/{}'.format(img_name)
            train_data.append([path_img,age,gender])
        else:
            path_img = 'test/{}'.format(img_name)
            test_data.append([path_img,age,gender])


for file_name in file_names:
    csv_flnm = '{}.csv'.format(file_name)
    with open(csv_flnm,'w',newline='',encoding ='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        if file_name=='test':
            writer.writerows(test_data)
        else:
            writer.writerows(train_data)
    print('{} csv file creation successfull'.format(file_name))


trn_df = pd.read_csv('dataset/train.csv')
val_df = pd.read_csv('dataset/test.csv')
# print(trn_df.head)

class data_maker(Dataset):
    def __init__(self, df):
        self.df = df
        self.normalize = NORMALIZE
    def __len__(self): return len(self.df)

    def __getitem__(self, ix):
      f = self.df.iloc[ix].squeeze()
      file = 'dataset/{}'.format(f.file)
      age = f.age
      gender = f.gender ##1 for female and 0 for male
      image = cv2.imread(file)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      return image, age, gender

    def preprocess_image(self, image):
        im =cv2.resize(image, (224, 224))
        im = torch.tensor(im).permute([2,0,1])
        im = self.normalize(im/255.0)
        return im[None]

    def collate_fn(self, batch):
        ims, ages, genders = [], [], []
        for image, age, gender in batch:
            genders.append(float(gender))
            ages.append(float(int(age)/116))
            im = self.preprocess_image(image)
            ims.append(im)
        ages, genders = [torch.tensor(x).to(device).float() for x in [ages, genders]]
        ims = torch.cat(ims).to(device)
        return ims, ages, genders

trn = data_maker(trn_df)
val = data_maker(val_df)

train_loader = DataLoader(trn, batch_size=32, shuffle=True,
                          drop_last=True, collate_fn=trn.collate_fn)
test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn)

# ims, ages, genders = next(iter(test_loader))
# print(ims.shape,ages.shape,genders.shape)

