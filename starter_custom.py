from basic_fcn import *
from dataloader import *
from custom_network import *
from utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy


#BB - set device choice
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#model save path:
FILE = 'custom_model.pth'

#BB - Batch size?
batchsize = 32

augment_data = False
train_dataset = TASDataset('tas500v1.1', augment_data=augment_data) 
val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')


# train_dataset = TASDataset('tas500v1.1') 
# val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
# test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')


train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

#BB - Changed epochs
epochs = 300 

#N epochs patience
epoch_no_imp = 0
epoch_stop = 10

# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
# The below function will return None by default, in order to get loss_weights, give weighting_method argument
loss_weights = get_loss_weights(weighting_method='basic')
# loss_weights = get_loss_weights()
if loss_weights is not None:
    loss_weights = loss_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights).to(device)#BB - Chose CEL
else:
    criterion = nn.CrossEntropyLoss().to(device) #BB - Chose CEL
n_class = 10
fcn_model = Res_Unet(n_class=n_class)
fcn_model.apply(init_weights)

#BB - Adding lr:
lr = 0.01


fcn_model = fcn_model.to(device) #transfer the model to the device

#BB - added optimizer
#optimizer = optim.Adam(fcn_model.parameters(), lr=0.01) # choose an optimizer
optimizer = optim.AdamW(fcn_model.parameters(), weight_decay = .001, lr=lr)

#BB - Adding lr scheduler:
#scheduler = ExponentialLR(optimizer, 0.9) ###added schedule




print(device)
def train():
    
    epoch_no_imp = 0
    epoch_stop = 10
    
    best_iou_score = 0.0
    best_acc = 0.0
    best_loss = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        losses = []
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            
            #BB - reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            outputs = fcn_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            #BB
            loss = criterion(outputs, labels) #calculate loss
            losses.append(loss.item())
            
            #BB - backpropagate
            loss.backward()

            #BB - update the weights
            optimizer.step()
        
        #Turn on if you want to mess with scheduling lr
        #scheduler.step()
            
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        train_losses.append(np.mean(losses))        

        current_miou_score, current_acc, current_loss = val(epoch)
        val_losses.append(current_loss)
        
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            best_acc = current_acc
            best_loss = current_loss
            epoch_no_imp = 0

            #Save to model
            torch.save(fcn_model.state_dict(), FILE)

        else:
          epoch_no_imp += 1

          if epoch_no_imp >= epoch_stop:
            print(f'No improvement after epoch {epoch-epoch_stop}')
            print(f'iou: {best_iou_score}, pixel acc: {best_acc}, loss: {best_loss}')
            break
    np.save('train_losses.npy',np.array(train_losses))
    np.save('validation_losses.npy',np.array(val_losses))
            
    

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            print(input.shape)
            output = fcn_model(input)

            print(output.shape)
            loss = criterion(output, label) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output, dim=1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util


    print(f"\nLoss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}\n")

    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(accuracy), np.mean(losses)

def test():

    #Load model
    fcn_model = Res_Unet(n_class=n_class)
    fcn_model.load_state_dict(torch.load(FILE))
    
    fcn_model = fcn_model.to(device)
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's
            output = fcn_model(input)

            loss = criterion(output, label) #calculate the loss
            losses.append(loss.item()) 

            pred = torch.argmax(output, dim=1) 
            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))
            accuracy.append(pixel_acc(pred, label))

    print(f"Loss is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc is {np.mean(accuracy)}")

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()