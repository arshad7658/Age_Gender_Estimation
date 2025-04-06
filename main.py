from torchvision import models
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
import time
import cv2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

trn_df = pd.read_csv('train_labels.csv')
val_df = pd.read_csv('val_labels.csv')
print(trn_df['age'].value_counts())

class data_maker(Dataset):
    def __init__(self, df):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    def __len__(self): return len(self.df)

    def __getitem__(self, ix):
      f = self.df.iloc[ix].squeeze()
      file = f.file
      age = f.age
      gender = f.gender == 'Female'
      image = cv2.imread(file)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      return image, age, gender

    def preprocess_image(self, image):
        im =cv2.resize(image, (224, 224))
        im = torch.tensor(im).permute([2,0,1])
        im = self.normalize(im/255.0)
        return im[None]

    def preprocess_age(self, age):
      if type(age)== str:
        if 'more' in age:
          return float(75/80)
        if '-' in age:
          age_range = age.split('-')
          age = (int(age_range[0])+int(age_range[1]))/2
          return float(age/80)
      return float(int(age)/80)

    def collate_fn(self, batch):
        ims, ages, genders = [], [], []
        for image, age, gender in batch:
            genders.append(float(gender))
            age_prep = self.preprocess_age(age)
            ages.append(age_prep)
            im = self.preprocess_image(image)
            ims.append(im)
        ages, genders = [torch.tensor(x).to(device).float() for x in [ages, gender]]
        ims = torch.cat(ims).to(device)
        return ims, ages, genders

trn = data_maker(trn_df)
val = data_maker(val_df)
train_loader = DataLoader(trn, batch_size=32, shuffle=True,
                          drop_last=True, collate_fn=trn.collate_fn)
test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn)

# ims, gens, ages = next(iter(train_loader))

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

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model.to(device), total_loss, optim

def train_batch(data, model, optim, criteria):
    model.train()
    optim.zero_grad()
    im, age, gender = data
    pred_gender, pred_age = model(im)
    gender_loss, age_loss = criteria
    age_L = age_loss(pred_age.squeeze(), age)
    gender_L = gender_loss(pred_gender.squeeze(), gender)
    total_L=gender_L+age_L
    total_L.backward()
    optim.step()
    return total_L

def val_batch(data, model, criteria):
    model.eval()
    im, age, gender = data
    with torch.no_grad():
        pred_gender, pred_age = model(im)
    gender_loss, age_loss = criteria
    age_L = age_loss(pred_age.squeeze(), age)
    gender_L = gender_loss(pred_gender.squeeze(), gender)
    total_L=gender_L+age_L
    gender_accuracy_step_1 = (pred_gender>0.5).squeeze()
    gender_accuracy = (pred_gender == gender).float().sum()
    age_error = (torch.abs(age - pred_age).float().sum())
    return total_L, gender_accuracy, age_error
val_gender_accuracies = []
val_age_maes = []
train_losses = []
val_losses = []

n_epochs = 5
best_test_loss = 1000
start = time.time()

model, loss_, opt = get_model()

for epoch in range(n_epochs):
    epoch_train_loss, epoch_test_loss = 0, 0
    val_age_mae, val_gender_acc, ctr = 0, 0, 0
    _n = len(train_loader)
    print('Current Epoch: {}'.format(epoch+1))
    for ix, data in enumerate(train_loader):
        loss = train_batch(data, model, opt, loss_)
        epoch_train_loss += loss.item()
        print('{} batch_done'.format(ix+1))

    for ix, data in enumerate(test_loader):
        loss, gender_acc, age_mae = val_batch(data, model, loss_)
        epoch_test_loss += loss.item()
        val_age_mae += age_mae
        val_gender_acc += gender_acc
        ctr += len(data[0])
        print('{} batch_done for test'.format(ix + 1))

    val_age_mae /= ctr
    val_gender_acc /= ctr
    epoch_train_loss /= len(train_loader)
    epoch_test_loss /= len(test_loader)

    elapsed = time.time()-start
    best_test_loss = min(best_test_loss, epoch_test_loss)
    print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, n_epochs, time.time()-start, (n_epochs-epoch)*(elapsed/(epoch+1))))
    info = f'''Epoch: {epoch+1:03d}\tTrain Loss: {epoch_train_loss:.3f}\tTest: {epoch_test_loss:.3f}\tBest Test Loss: {best_test_loss:.4f}'''
    info += f'\nGender Accuracy: {val_gender_acc*100:.2f}%\tAge MAE: {val_age_mae:.2f}\n'
    print(info)

    val_gender_accuracies.append(val_gender_acc)
    val_age_maes.append(val_age_mae)

torch.save(model.state_dict(), 'trained.pth')

epochs = np.arange(1,len(val_gender_accuracies)+1)
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.flat
ax[0].plot(epochs, [v.item() for v in val_gender_accuracies], 'bo')
ax[1].plot(epochs, [v.item() for v in val_age_maes], 'r')
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[1].set_ylabel('MAE')
ax[0].set_title('Validation Gender Accuracy')
ax[0].set_title('Validation Age Mean-Absolute-Error')
plt.show()