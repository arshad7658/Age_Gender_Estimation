import torch
from model_architecture import device,model
from data_preparation import NORMALIZE
import cv2
import numpy as np


def predictions(img_path):
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict)
    model.eval()
    # input_img=cv2.imread(img_path)
    img =cv2.resize(img_path, (224, 224))
    img = torch.tensor(img).permute([2,0,1])
    im = NORMALIZE(img/255.0)
    im = im[None]
    im = im.to(device)
    age, gender = model(im)
    gender = gender.to('cpu').detach().numpy()
    age = (age.to('cpu').detach().numpy())
    
    pred_gender = 'Male' if int(gender[0][0]>0.5) == 1 else 'Female'
    pred_age = round(age[0][0]*80)                           
    

    return pred_gender,pred_age
