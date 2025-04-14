from training_and_validating import train
from predict import predictions
import os

# train()

img_path = os.path.join('face.jpg')


gender,age = predictions(img_path = img_path)

print(gender)