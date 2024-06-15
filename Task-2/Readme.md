# Task-2
## Convolutional Block Attention Module (CBAM)
The Convolutional Block Attention Module (CBAM) enhances convolutional neural networks by focusing on important features within the data.
It operates within each convolutional block, analyzing features across two dimensions: channels and spatial location. 
By applying attention mechanisms, CBAM assigns weights to different features. 
This allows the network to prioritize informative features and suppress irrelevant ones, leading to more refined feature maps. This focus on key information can improve the accuracy of tasks like image classification and object detection.


Implementation for CBAM block can be found in `CBAM.py`.

## Squeeze And Excitation
Squeeze-and-Excitation Networks (SENets) improve the performance of convolutional neural networks (CNNs) by dynamically adjusting feature importance.
Unlike standard CNNs, SENets don't just focus on spatial relationships within features.
They include a special "squeeze-and-excitation" block that analyzes the importance of each feature channel (think of channels as different feature maps).
This allows the network to selectively emphasize informative channels and suppress less important ones. 
This dynamic weighting leads to better feature representation and improved performance in tasks like image recognition and object detection.


Implementation for Squeeze and Excitation block can be found in `SqueezeAndExcitation.py`.

## Image Segmentation
I chose image segmentation task using `Cityscapes` dataset. I have incorporated these blocks as part of Unet architecture to perform the segmentation task. 
The corresponding unet architecture implementations can be found in `CBAM_Unet.py` and `SE_Unet.py`.