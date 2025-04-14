import torch
from model_architecture import device,model
from data_preparation import NORMALIZE
import cv2
import numpy as np


def predictions(img_path):
    state_dict = torch.load('trained_weights/best_model_weights3.pth')
    model.load_state_dict(state_dict)
    model.eval()
    input_img=cv2.imread(img_path)
    img =cv2.resize(input_img, (224, 224))
    img = torch.tensor(img).permute([2,0,1])
    im = NORMALIZE(img/255.0)
    im = im[None]
    im = im.to(device)
    age, gender = model(im)
    gender = gender.to('cpu').detach().numpy()
    age = (age.to('cpu').detach().numpy())
    
    pred_gender = np.where(gender[0][0]<0.5, 'Male', 'Female')
    pred_age = int(age[0][0]*116*(10**12))                           
    

    return pred_gender,pred_age