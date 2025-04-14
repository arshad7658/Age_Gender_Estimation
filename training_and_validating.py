import matplotlib.pyplot as plt
import torch
import time
from data_preparation import train_loader, test_loader
from model_architecture import model,loss_,opt
import numpy as np


def train_batch(data, model, optim, loss_function):
    model.train()
    im, age, gender = data
    optim.zero_grad()
    pred_gender, pred_age = model(im)
    gender_loss_function, age_loss_function = loss_function
    age_loss = age_loss_function(pred_age.squeeze(), age)
    gender_loss = gender_loss_function(pred_gender.squeeze(), gender)
    total_loss=gender_loss+age_loss
    total_loss.backward()
    optim.step()
    return total_loss

def val_batch(data, model, loss_function):
    model.eval()
    im, age, gender = data
    with torch.no_grad():
        pred_gender, pred_age = model(im)
    gender_loss, age_loss = loss_function
    age_L = age_loss(pred_age.squeeze(), age)
    gender_L = gender_loss(pred_gender.squeeze(), gender)
    total_L=gender_L+age_L
    gender_accuracy_step_1 = (pred_gender>0.5).squeeze()
    gender_accuracy = (gender_accuracy_step_1 == gender).float().sum()
    age_error = (torch.abs(age - pred_age).float().sum())
    return total_L, gender_accuracy, age_error


def train():
    val_gender_accuracies = []
    val_age_maes = []
    train_losses = []
    val_losses = []

    n_epochs = 5
    best_test_loss = 1000
    start = time.time()

    for epoch in range(n_epochs):
        epoch_train_loss, epoch_test_loss = 0, 0
        val_age_mae, val_gender_acc, ctr = 0, 0, 0
        _n = len(train_loader)
        print('Current Epoch: {}'.format(epoch+1))
        for ix, data in enumerate(train_loader):
            loss = train_batch(data, model, opt, loss_)
            epoch_train_loss += loss.item()
        print('Current Epoch: {} done for training'.format(epoch+1))

        for ix, data in enumerate(test_loader):
            loss, gender_acc, age_mae = val_batch(data, model, loss_)
            epoch_test_loss += loss.item()
            val_age_mae += age_mae
            val_gender_acc += gender_acc
            ctr += len(data[0])
        print('Current Epoch: {} done for testing'.format(epoch+1))

        val_age_mae /= ctr
        val_gender_acc /= ctr
        epoch_train_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)

        best_test_loss = min(best_test_loss, epoch_test_loss)

        if epoch_test_loss<epoch_train_loss:
            print('Validation loss improved. Saving model.')
            torch.save(model.state_dict(), 'best_model_weights.pth')

        torch.save(model.state_dict(), 'best_model_weights{}.pth'.format(epoch))

        elapsed = time.time()-start
        print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, n_epochs, time.time()-start, (n_epochs-epoch)*(elapsed/(epoch+1))))
        info = f'''Epoch: {epoch+1:03d}\tTrain Loss: {epoch_train_loss:.3f}\tTest: {epoch_test_loss:.3f}\tBest Test Loss: {best_test_loss:.4f}'''
        info += f'\nGender Accuracy: {val_gender_acc*100:.2f}%\tAge MAE: {val_age_mae:.2f}\n'
        
        print(info)

        val_gender_accuracies.append(val_gender_acc)
        val_age_maes.append(val_age_mae)




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