# CSI-Authentication
：Source code for the paper CSI-Based Physical Layer Authentication via Deep Learning
:This repository contains the code for the paper CSI-Based Physical Layer Authentication via Deep Learning. In this work, we propose a newly deep-CSI-based authentication scheme in wireless communication. We map CSI to a device’s location and further to its authenticated identity via deep learning in a static environment. Therefore, the proposed scheme does not require the cooperation of cryptography-based authentication to achieve initial authentication. The deep-learning-based authenticator with a confidence score branch is designed to learn the mapping relationship between the CSI and the identity. The confidence score branch can output a scalar that indicates whether the device is legitimate or not in the absence of illegitimate device CSI samples. CSI data are constructed as CSI images and implementation tricks are proposed to train the authenticator. 

## Dependencies
PyTorch v0.3.0<br>
tqdm<br>
visdom<br>
seaborn<br>
Pillow<br>
scikit-learn<br>

## Dataset
`channel_ind.mat` in folder `data` is the legitimate CSI samples dataset. `channel_ood.mat` in folder `data` is the illegitimate CSI samples dataset. `channel_10percent.mat`, `channel_30percent.mat` and `channel_60percent.mat` are legitimate CSI samples datasets with no channel estimation error, 10% channel estimation error, 30% channel estimation error and 60% channel estimation error, respectively. <br>
To evaluate the performance of the deep-learning-based authenticator, Quasi-Deterministic Radio Channel Generator (QuaDRiGa) is used to simulate multipath fading channels and obtain CSI data. QuaDRiGa generates realistic radio channel impulse responses for system-level simulations. We randomly select 10 legitimate devices with different positions for the training dataset and sampled 5000 CSI images for each device. For the test dataset, each legitimate device samples 1000 CSI images, and 10 illegitimate devices are randomly selected and each with 1000 CSI images.

## Training
Train a model with a confidence score branch with `train.py`. Training logs will be stored in the logs/ folder, while checkpoints are stored in the checkpoints/ folder.  We use VGG11 added with a confidence score branch to evaluate the proposed scheme. The authenticator is trained using stochastic gradient descent, with Nesterov momentum of 0.9. We train the authenticator for 36 epochs with a batch size of 180 and a weight decay of 2e-4. The initial learning rate is 0.1 and reduced by a factor of 5× after the 6th, 12th, 18th, 24th, 30th epochs.

## Anomaly Detection
Evaluate a trained model with `anomaly_detection.py`. False alarm rate (FAR), miss detection error (MDR), detection error (DE), accuracy (ACC), macro-averaged F1 score (Macro F1-score), confidence scores density distribution, confusion matrix are automatically saved.
