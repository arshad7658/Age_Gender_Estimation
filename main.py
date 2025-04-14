from torchvision import models
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import csv
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# print(device)


# # def make_csv_for_utk_data():
# #     test_dir= os.path.join('test')
# #     train_dir = os.path.join('train')
# #     data_dirs = [test_dir,train_dir]
# #     file_names = ['test','train']
# #     header = ['filepath','age', 'gender']
# #     train_data,test_data=[],[]
# #     for idx,data_dir in enumerate(data_dirs):
# #         for img_name in os.listdir(data_dir):
# #             age  = img_name.split('_')[0]
# #             gender = img_name.split('_')[1] 
# #             if idx==1:
# #                 path_img = 'train/{}'.format()
# #                 train_data.append([path_img,age,gender])
# #             else:
# #                 path_img = 'test/{}'.format()
# #                 test_data.append([path_img,age,gender])


# #     for file_name in file_names:
# #         csv_flnm = '{}.csv'.format(file_name)
# #         with open(file_name,'w',newline='',encoding ='utf-8') as file:
# #             writer = csv.writer(file)
# #             writer.writerow(header)
# #             if file_name=='test':
# #                 writer.writerows(test_data)
# #             else:
# #                 writer.writerows(train_data)
# #         print('{} csv file creation successfull'.format(file_name))

# # make_csv_for_utk_data()


trn_df = pd.read_csv('dataset/train.csv')
val_df = pd.read_csv('dataset/test.csv')
print(trn_df.head)

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

# class data_maker(Dataset):
#     def __init__(self, df):
#         self.df = df
#         self.normalize = NORMALIZE
#     def __len__(self): return len(self.df)

#     def __getitem__(self, ix):
#       f = self.df.iloc[ix].squeeze()
#       file = 'dataset/{}'.format(f.file)
#       age = f.age
#       gender = f.gender ##1 for female and 0 for male
#       image = cv2.imread(file)
#       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#       return image, age, gender

#     def preprocess_image(self, image):
#         im =cv2.resize(image, (224, 224))
#         im = torch.tensor(im).permute([2,0,1])
#         im = self.normalize(im/255.0)
#         return im[None]

#     def collate_fn(self, batch):
#         ims, ages, genders = [], [], []
#         for image, age, gender in batch:
#             genders.append(float(gender))
#             ages.append(float(int(age)/116))
#             im = self.preprocess_image(image)
#             ims.append(im)
#         ages, genders = [torch.tensor(x).to(device).float() for x in [ages, genders]]
#         ims = torch.cat(ims).to(device)
#         return ims, ages, genders

# trn = data_maker(trn_df)
# val = data_maker(val_df)


# train_loader = DataLoader(trn, batch_size=32, shuffle=True,
#                           drop_last=True, collate_fn=trn.collate_fn)
# test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn)

# ims, ages, genders = next(iter(test_loader))

# print(ims.shape,ages.shape,genders.shape)


def get_model():
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512,512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    class age_gender_classifier(nn.Module):
        def __init__(self):
            super(age_gender_classifier, self).__init__()
            self.intermediate = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                )

            self.age_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
                )
            self.gender_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
                )

        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)
            return age, gender

    model.classifier = age_gender_classifier()

    age_loss = nn.L1Loss()
    gender_loss = nn.BCELoss()

    total_loss = age_loss, gender_loss

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model.to(device), total_loss, optim

model, loss_, opt = get_model()

# print(model)

# def train_batch(data, model, optim, loss_function):
#     model.train()
#     im, age, gender = data
#     optim.zero_grad()
#     pred_gender, pred_age = model(im)
#     gender_loss_function, age_loss_function = loss_function
#     age_loss = age_loss_function(pred_age.squeeze(), age)
#     gender_loss = gender_loss_function(pred_gender.squeeze(), gender)
#     total_loss=gender_loss+age_loss
#     total_loss.backward()
#     optim.step()
#     return total_loss

# def val_batch(data, model, loss_function):
#     model.eval()
#     im, age, gender = data
#     with torch.no_grad():
#         pred_gender, pred_age = model(im)
#     gender_loss, age_loss = loss_function
#     age_L = age_loss(pred_age.squeeze(), age)
#     gender_L = gender_loss(pred_gender.squeeze(), gender)
#     total_L=gender_L+age_L
#     gender_accuracy_step_1 = (pred_gender>0.5).squeeze()
#     gender_accuracy = (gender_accuracy_step_1 == gender).float().sum()
#     age_error = (torch.abs(age - pred_age).float().sum())
#     return total_L, gender_accuracy, age_error

# val_gender_accuracies = []
# val_age_maes = []
# train_losses = []
# val_losses = []

# n_epochs = 5
# best_test_loss = 1000
# start = time.time()

# for epoch in range(n_epochs):
#     epoch_train_loss, epoch_test_loss = 0, 0
#     val_age_mae, val_gender_acc, ctr = 0, 0, 0
#     _n = len(train_loader)
#     print('Current Epoch: {}'.format(epoch+1))
#     for ix, data in enumerate(train_loader):
#         loss = train_batch(data, model, opt, loss_)
#         epoch_train_loss += loss.item()
#     print('Current Epoch: {} done for training'.format(epoch+1))

#     for ix, data in enumerate(test_loader):
#         loss, gender_acc, age_mae = val_batch(data, model, loss_)
#         epoch_test_loss += loss.item()
#         val_age_mae += age_mae
#         val_gender_acc += gender_acc
#         ctr += len(data[0])
#     print('Current Epoch: {} done for testing'.format(epoch+1))

#     val_age_mae /= ctr
#     val_gender_acc /= ctr
#     epoch_train_loss /= len(train_loader)
#     epoch_test_loss /= len(test_loader)

#     best_test_loss = min(best_test_loss, epoch_test_loss)

#     if epoch_test_loss<epoch_train_loss:
#         print('Validation loss improved. Saving model.')
#         torch.save(model.state_dict(), 'best_model_weights.pth')

#     torch.save(model.state_dict(), 'best_model_weights{}.pth'.format(epoch))

#     elapsed = time.time()-start
#     print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, n_epochs, time.time()-start, (n_epochs-epoch)*(elapsed/(epoch+1))))
#     info = f'''Epoch: {epoch+1:03d}\tTrain Loss: {epoch_train_loss:.3f}\tTest: {epoch_test_loss:.3f}\tBest Test Loss: {best_test_loss:.4f}'''
#     info += f'\nGender Accuracy: {val_gender_acc*100:.2f}%\tAge MAE: {val_age_mae:.2f}\n'
#     print(info)

#     val_gender_accuracies.append(val_gender_acc)
#     val_age_maes.append(val_age_mae)




# epochs = np.arange(1,len(val_gender_accuracies)+1)
# fig,ax = plt.subplots(1,2,figsize=(10,5))
# ax = ax.flat
# ax[0].plot(epochs, [v.item() for v in val_gender_accuracies], 'bo')
# ax[1].plot(epochs, [v.item() for v in val_age_maes], 'r')
# ax[0].set_xlabel('Epochs')
# ax[1].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy')
# ax[1].set_ylabel('MAE')
# ax[0].set_title('Validation Gender Accuracy')
# ax[0].set_title('Validation Age Mean-Absolute-Error')
# plt.show()


model,_,_= get_model()
model.to(device)
# path_to_state = os.path.join('trained.pth')
state_dict = torch.load('best_model_weights3.pth')
model.load_state_dict(state_dict)
model.eval()

img_path = os.path.join('face.jpg')


def predict_gender_and_age(img_path):
    input_img=cv2.imread(img_path)
    img =cv2.resize(input_img, (224, 224))
    img = torch.tensor(img).permute([2,0,1])
    im = NORMALIZE(img/255.0)
    im = im[None]
    im = im.to(device)
    age, gender = model(im)
    gender = gender.to('cpu').detach().numpy()
    age = (age.to('cpu').detach().numpy())
    plt.imshow(input_img)
    # plt.figure(244,244)
    plt.axis('off')
    cv2.imshow('Image', input_img)
    plt.show()
    print(f'Gender : {np.where(gender[0][0]<0.5, 'Male', 'Female')} ; Age: {int(age[0][0]*116*(10**12))}')

predict_gender_and_age(img_path)
