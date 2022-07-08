import numpy as np
import torch
from torch._C import dtype
from torchvision import transforms
from torch.nn import functional as func
import matplotlib.pyplot as plt


def get_loss_weights(weighting_method=None):
# Get the loss according to different weighting methods, output will be tensor if not None
  if weighting_method is not None:
    sample_per_class = np.load('sample_per_class.npy')
    print("weighting-method: {}".format(weighting_method))
  if weighting_method == 'INS':
    loss_weights = ins_loss_weights(sample_per_class)
  elif weighting_method == 'ISNS':
    loss_weights = ins_loss_weights(sample_per_class, power=0.5)
  elif weighting_method == 'basic':
    loss_weights = basic_loss_weights(sample_per_class)
  else:
    return None
  return loss_weights


def ins_loss_weights(sample_per_class, power=1):
  loss_weights = np.power(sample_per_class, power)
  loss_weights = np.sum(loss_weights)/loss_weights
  loss_weights = loss_weights / 100
  loss_weights = torch.from_numpy(loss_weights)
  return loss_weights 


def basic_loss_weights(sample_per_class):
  loss_weights = sample_per_class/sample_per_class.sum()
  loss_weights = 1 - loss_weights
  loss_weights = torch.from_numpy(loss_weights)
  return loss_weights


def iou(pred, target, n_classes = 10):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for undefined class ("9")
  for cls in range(n_classes-1):  # last class is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    
    #BB - implemented intersection/union
    intersection = (pred_inds*target_inds).sum().item()
    union = pred_inds.sum().item() + target_inds.sum().item() - intersection
    
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(intersection/union)

  return np.array(ious)
  

def pixel_acc(pred, target, n_classes = 10):
  
  #BB - Keep track of total count
  correct = 0
  total = 0

  #Reshape
  pred = pred.view(-1)
  target = target.view(-1)

  for cls in range(n_classes-1):

    #BB - Identify preds and targets
    pred_inds = pred == cls
    target_inds = target == cls
    
    #BB - Add to correct if correct, and add to total samps
    correct += (pred_inds*target_inds).sum().item()
    total += target_inds.sum().item()
  
  #BB - Return fraction
  return correct/total

def dice_loss(preds, target, eps=1e-7):

    num_classes = preds.shape[1]

    #One hot encoding of classes
    hot = torch.eye(num_classes)[target.squeeze(1)]
    
    #Alter order
    hot = hot.permute(0, 3, 1, 2).float()
    probs = func.softmax(preds, dim=1)
    
    #Set to same type as predictions
    hot = hot.type(preds.type())
    
    #Calculate dimentions
    dims = (0,) + tuple(range(2, target.ndimension()))
    
    #Calculate intersection and union
    intersection = torch.sum(probs * hot, dims)
    union = torch.sum(probs + hot, dims)

    #Calculate dice score
    dice = (2. * intersection / (union + eps)).mean()
    
    #Return dice loss
    return (1 - dice)


def visualize_prediction(image, prediction):
  invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
  class2color = {
                #terrain
                0 : (192,192,192), \
                #vegatation
                1 : (60,179,113), \
                #construction
                2 : (0,132,111), \
                #vehicle
                3: (0,0,142), \
                #sky
                4: (135,206,250), \
                #object
                5: (128,0,128), \
                #human
                6: (220, 20, 60), \
                #animal
                7: ( 255,182,193),\
                #void
                8: (220,220,220), \
                #undefined
                9: (0,  0,  0)    
  }
  img_copy = torch.detach(image).clone()
  img_copy = img_copy.cpu()
  img_copy = invTrans(img_copy)
  img_copy = img_copy.permute(1,2,0).numpy()
  pred_copy = torch.detach(prediction).clone()
  pred_copy = pred_copy.cpu()
  pred_copy = pred_copy.numpy()

  segmentation_mask = np.zeros((pred_copy.shape[0],pred_copy.shape[1],3), dtype=int)
  for c in range(10):
    pred_ind = pred_copy == c
    segmentation_mask[pred_ind] = class2color[c]

  plt.figure()
  plt.imshow(img_copy, interpolation='none')
  plt.imshow(segmentation_mask, alpha=0.5, interpolation='none')
  plt.savefig('segmented_test_img.png')

  