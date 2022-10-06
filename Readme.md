Image Segmentation: A U-NET Implementation
------------------------------------------

This notebook provides a PyTorch implementation of the U-Net for image segmentation. The notebook can be run by changing the 'Data_PATH' variable to a directory containing a 'train' folder for images, 'train_labels' for pixel labels, 'test' for test images, and 'test_labels' for test pixel labels. The model and training loop are standard PyTorch implementations and can be repurposed to other image segmentation problems. The Utils and Dataloader are specific to the TAS500 Dataset (https://mucar3.de/icpr2020-tas500/), a segmentation dataset for autonomous driving in unstructured environments.
