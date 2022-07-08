from dataloader import *
import PIL

# Don't forget to run this again if you change input image size!
# We will only need this once

# train_dataset = TASDataset('tas500v1.1') 
train_dataset = TASDataset('drive/MyDrive/tas500v1.1') 


paths = train_dataset.paths
sample_per_class = np.zeros(10)

for ind in range(len(paths)):
    mask_image = np.asarray(PIL.Image.open(paths[ind][1]).resize((train_dataset.width, train_dataset.height), PIL.Image.NEAREST))
    mask =  rgb2vals(mask_image, train_dataset.color2class)
    for c in range(10):
        sample_per_class[c] = sample_per_class[c] + np.sum(mask==c)

# Save the number of samples per class.
np.save('sample_per_class.npy', sample_per_class)
