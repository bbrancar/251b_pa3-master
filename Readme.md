CSE 251B - Programming Assignment 3 Readme
------------------------------------------
In order to make any model run in this project, you need to have dataset folder in the same folder with the other files.
utils.py includes the performance measurement, segmentation visualization and some other intermediate functions.
graph.py includes graphing function using training and validation losses.
dataloader.py includes dataloading functions.
get_samples_per_class.py file iterates through the training set and calculates number of pixels per class. Later on, this
sample_per_class.npy file is used for generating class weights.

NOTE: If you do not have sample_per_class.npy file in the same folder with project files. The code will not work with weighted
loss. User needs to run get_samples_per_class.py to get that numpy array. It is recommended to run that python file before
moving on.

-Baseline Model / Improved Baseline Model
For the baseline model and its improved version, you need to run starter.py file. Note that in that file, there are couple
parameter that effects training, such as learning_rate, batch_size, epochs, early_stop_epoch. User can change those to
experiment with the baseline model. Model architecture can be found in basic_fcn.py

If user wants to use data augmentation, they need to enable it by setting augment_data = True in starter.py.
If user wants to use weighted loss, they need to set weighting_method in get_loss_weights line in starter.py. By default,
the training is done without weighted loss. User can set weighting_method as 'basic', 'INS' or 'ISNS'.

-Custom Architecture
For this model to work, user needs to run starter_custom.py file. Model architecture can be found in custom_model.py Similar 
to previous part, user can change the parameters such as learning_rate, batch_size, epochs, early_stop, augment_data and 
weighting_method to play with the training.
 
-Transfer Learning
Architecture of transfer learning model can be found in res34_fcn.py. If user wants to try this model, they can follow
transfer_learning.ipynb notebook. Just be careful about the dataset path, user should change that variable in the notebook
accordingly.

-UNet Architecture
The architecture of UNet model can be found in unet.py file. In order to train and/or test the model, user should run
run_unet.py file. Similar to part 1 user can change the parameters such as learning_rate, batch_size, epochs, early_stop, 
augment_data and to play with the training. In this part, dice loss is used instead of weighted loss.
