from torchvision import models
import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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