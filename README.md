# CSI-Authentication
Source code for the paper CSI-Based Physical Layer Authentication via Deep Learning
This repository contains the code for the paper CSI-Based Physical Layer Authentication via Deep Learning. In this work, we propose a newly deep-CSI-based authentication scheme in wireless communication. We map CSI to a device’s location and further to its authenticated identity via deep learning in a static environment. Therefore, the proposed scheme does not require the cooperation of cryptography-based authentication to achieve initial authentication. The deep-learning-based authenticator with a confidence score branch is designed to learn the mapping relationship between the CSI and the identity. The confidence score branch can output a scalar that indicates whether the device is legitimate or not in the absence of illegitimate device CSI samples. CSI data are constructed as CSI images and implementation tricks are proposed to train the authenticator. 
# Dependencies
PyTorch v0.3.0
tqdm
visdom
seaborn
Pillow
scikit-learn
