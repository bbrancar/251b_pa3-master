from unet import *
from dataloader import *
from utils import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #For GPU
FILE = 'unet_model.pth'

#BB - Batch size?
batchsize = 4

#CHANGE THIS TO YOUR PATH!
Data_PATH = 'tas500v1.1'

augment_data = False
train_dataset = TASDataset(Data_PATH, augment_data=augment_data) 
val_dataset = TASDataset(Data_PATH, eval=True, mode='val')
test_dataset = TASDataset(Data_PATH, eval=True, mode='test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

#BB - Changed epochs
epochs = 1000

#N epochs patience
epoch_no_imp = 0
epoch_stop = 30

#Use dice loss
criterion = dice_loss
n_class = 10
unet_model = UNET(n_class=n_class)
unet_model.apply(init_weights)

#print('initialized model')
#mem_report()

#BB - added optimizer
optimizer = optim.Adam(unet_model.parameters(), lr=0.0009) ### ADAM

#Put model on device
unet_model = unet_model.to(device) #transfer the model to the device

#Check that running GPU
print(f'currently on GPU? {next(unet_model.parameters()).is_cuda}')

#Train
def train():

    best_iou_score = 0.0
    best_acc = 0
    best_loss = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        ts = time.time()
        
        for iter, (inputs, labels) in enumerate(train_loader):
            
            losses = []
            #Change format
            labels = labels.long()

            #eset optimizer gradients
            optimizer.zero_grad()
              
            # Train
            inputs = inputs.to(device)
            labels = labels.to(device)           
            outputs = unet_model(inputs)
            loss = criterion(outputs, labels) #calculate loss
            loss.backward()
            optimizer.step()

            #Save loss
            loss_cpu = loss.to("cpu")
            losses.append(loss_cpu.item())
            if iter % 10 == 0:
                print("epoch {}, iter {}, loss: {}".format(epoch, iter, loss_cpu.item()))

        #Append to train losses
        train_losses.append(np.mean(np.array(losses)))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        

        current_miou_score, current_acc, current_loss = val(epoch)
        val_losses.append(current_loss)
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            best_acc = current_acc 
            best_loss = current_loss

            epoch_no_imp = 0

            #Save to model folder
            torch.save(unet_model.state_dict(), FILE)
      
        else:
          epoch_no_imp += 1
          print(f'No improvements for {epoch_no_imp} epochs.')

          if epoch_no_imp >= epoch_stop:
            print(f'No improvement after epoch {epoch-epoch_stop}') 
            print(f'iou: {best_iou_score}, pixel acc: {best_acc}, loss: {best_loss}') ###ADD
            break
    
    return train_losses, val_losses

def val(epoch):
    unet_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):

            #Change format
            label = label.long()
            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.to(device) #transfer the labels to the same device as the model's

            output = unet_model(input)
            loss = criterion(output, label)
            loss_cpu = loss.to("cpu")

            losses.append(loss_cpu.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output, dim=1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util


    print(f"\nLoss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}\n")

    unet_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(accuracy), np.mean(losses)

def test():

    #Load model
    unet_model = UNET(n_class=n_class)
    unet_model.load_state_dict(torch.load(FILE))
    unet_model = unet_model.to(device) #transfer the model to the device
    inputs = []
    outputs = []
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):

            #Change format
            label = label.long()

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.to(device) #transfer the labels to the same device as the model's

            output = unet_model(input)

            loss = criterion(output, label) #calculate the loss
            losses.append(loss.item()) 

            pred = torch.argmax(output, dim=1)
            inputs.append(input)
            outputs.append(pred) 
            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))
            accuracy.append(pixel_acc(pred, label))

    print(f"Loss is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc is {np.mean(accuracy)}")
    return inputs, outputs
