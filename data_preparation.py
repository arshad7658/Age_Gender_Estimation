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


# test_dir= os.path.join('dataset','test')
# train_dir = os.path.join('dataset','train')
# data_dirs = [test_dir,train_dir]
# file_names = ['test','train']
# header = ['file','age', 'gender']
# train_data,test_data=[],[]
# for idx,data_dir in enumerate(data_dirs):
#     for img_name in os.listdir(data_dir):
#         age  = img_name.split('_')[0]
#         gender = img_name.split('_')[1] 
#         if idx==1:
#             path_img = 'train/{}'.format(img_name)
#             train_data.append([path_img,age,gender])
#         else:
#             path_img = 'test/{}'.format(img_name)
#             test_data.append([path_img,age,gender])


# for file_name in file_names:
#     csv_flnm = '{}.csv'.format(file_name)
#     with open(csv_flnm,'w',newline='',encoding ='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(header)
#         if file_name=='test':
#             writer.writerows(test_data)
#         else:
#             writer.writerows(train_data)
#     print('{} csv file creation successfull'.format(file_name))


trn_df = pd.read_csv('dataset/train.csv')
val_df = pd.read_csv('dataset/val.csv')
# print(trn_df.head)

class Datamaker(Dataset):
    def __init__(self,df):
        self.df = df
    def __len__(self):return len(self.df)
    def __getitem__(self, index):
        data = self.df.iloc[index].squeeze()
        file = 'dataset/{}'.format(data.file)
        age = data.age
        gender = 0 if data.gender == 'Male' else 1
        img = cv2.imread(file)
        # cv2.imshow('win', img)
        # cv2.destroyWindow('win')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img,age,gender
    def preprocess_image(self,img):
        img = cv2.resize(img,(224,224))
        img = torch.tensor(img).permute(2,0,1)
        img = NORMALIZE(img/255.0)
        return img[None]
    def collate_fn(self,batch):
        imgs,ages,genders = [ ],[ ],[ ]
        for img,age,gender in batch:
            img = self.preprocess_image(img)
            imgs.append(img)
            ages.append(float(age)/80)
            genders.append(float(gender))
        ages,genders = [torch.tensor(x).to(device).float() for x in [ages,genders]]
        imgs = torch.cat(imgs).to(device)
        return imgs,ages,genders



# class Datamaker(Dataset):
#     def __init__(self,df):
#         self.df = df
#     def __len__(self):return len(self.df)
#     def __getitem__(self,index):
#         data = self.df.iloc[index].squeeze()
#         age = data.age
#         gender = data.gender == 'Male'
#         file = f'dataset/{data.file}'
#         img = cv2.imread(file)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img, age, gender
#     def preprocess_image(self,img):
#         im = cv2.resize(img,(224,224))
#         im = torch.tensor(im).permute(2,0,1)
#         im = NORMALIZE(im/255.0)
#         return im[None]
#     def collate_fn(self,data):
#         img_list,age_list,gender_list = [ ], [ ], [ ]
#         for img,age,gender in data:
#             img = self.preprocess_image(img)
#             img_list.append(img)
#             age_list.append(float(age)/80)
#             gender_list.append(float(gender))
#         age_list,gender_list = [torch.tensor(x).float().to(device) for x in [age_list,gender_list]]
#         ims = torch.cat(img_list).to(device)
#         return ims,age_list,gender_list

trn = Datamaker(trn_df)
val = Datamaker(val_df)

train_loader = DataLoader(trn, batch_size=32, shuffle=True,
                          drop_last=True, collate_fn=trn.collate_fn)
test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn)

ims, ages, genders = next(iter(test_loader))

print(ims.shape,ages.shape,genders.shape)
