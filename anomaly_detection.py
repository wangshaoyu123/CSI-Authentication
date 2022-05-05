import pdb
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from sklearn import metrics      #机器学习工具，全流程API
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.autograd import Variable
import scipy.io as io
import seaborn as sns     #画图工具

from test_CSI_data_load import TestCSIDataLoad
from train_CSI_data_load import TrainCSIDataLoad
from models.vgg import VGG
from models.cnn import CNN
from utils.ood_metrics import tpr95, detection, far_threshhold, md_threshhold, roc, acc, predict
from utils.datasets import GaussianNoise, UniformNoise
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

ind_options = ['NLOS', 'LOS']
ood_options = ['NLOS_anomaly',
               'LOS_anomaly',
               'Uniform',
               'Gaussian',
               'all']
model_options = ['cnn', 'vgg']
process_options = ['confidence', 'confidence_scaling']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--ind_dataset', default='ind', choices=ind_options)
parser.add_argument('--ood_dataset', default='ood', choices=ood_options)
parser.add_argument('--model', default='vgg', choices=model_options)
parser.add_argument('--process', default='confidence_scaling', choices=process_options)
parser.add_argument('--batch_size', type=int, default=180)
parser.add_argument('--T', type=float, default=1000., help='Scaling temperature')
parser.add_argument('--epsilon', type=float, default=0.001, help='Noise magnitude')
parser.add_argument('--checkpoint', default='2022mainK_1_seed_0', type=str,
                    help='filepath for checkpoint to load')

args = parser.parse_args()
cudnn.benchmark = True  # Should make training should go faster for large models

filename = args.checkpoint    #后续用于加载对应的模型参数文件


transform = transforms.Compose([transforms.ToTensor()])

### 编写ind_dataset
### 编写ood_dataset
### 编写ind_loader
### 编写ood_loader
#读取CSI数据
CSIDataFileLoadOod = h5py.File('data\\channel_ood.mat','r') # 读取mat文件
CSIDataOod = np.transpose(CSIDataFileLoadOod['channel_ood'][:])#转置后才能和channel的维度一致
CSIDataOod = CSIDataOod.astype(np.float32)

CSIDataFileLoadInd = h5py.File('data\\channel_ind.mat','r') #读取mat文件
CSIDataInd = np.transpose(CSIDataFileLoadInd['channel_ind'][:])#转置后才能和channel的维度一致
CSIDataInd = CSIDataInd.astype(np.float32)

# permutation = np.random.permutation(CSIDataInd.shape[3])    
# CSIDataInd = CSIDataInd[:,:,:,permutation]
# CSIDataInd = CSIDataInd[:,:,:,0:10000]

CSIDataInd = CSIDataInd[:,:,:,0::5]
CSILable_ind = np.zeros(50000)
for i in range(0,10):
      CSILable_ind[0+5000*i:5000+5000*i] = i*np.ones(5000)
CSILable_ind = CSILable_ind[0::5]

#无意义的标签，为了使用CSIDataLoad
# CSILable_ind = np.zeros(50000)
# for i in range(0,10):
#     CSILable_ind[0+5000*i:5000+5000*i] = i*np.ones(5000)
# CSILable_ind = torch.from_numpy(CSILable_ind)
# CSILable_ind = CSILable_ind[permutation]
# CSILable_ind = CSILable_ind[0:10000]

#CSIDataOod = CSIDataOod[:,:,:,0::10]



#无意义的标签，为了使用CSIDataLoad
CSILable_ood = np.zeros(10000)

#数据集生成
ood_dataset = TestCSIDataLoad(CSIDataOod, CSILable_ood, transform)
ind_dataset = TestCSIDataLoad(CSIDataInd, CSILable_ind, transform)

num_classes = 10

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=0)

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=0)


##############################
### Load pre-trained model ###
##############################

if args.model == 'vgg':
    cnn = VGG("VGG11", num_classes=num_classes).cuda()
elif args.model == 'cnn':
    cnn = CNN(num_classes=num_classes).cuda()



#加载训练好的模型参数
model_dict = cnn.state_dict()
pretrained_dict = torch.load('checkpoints/' + filename + '.pt')
cnn.load_state_dict(pretrained_dict)
cnn = cnn.cuda()
cnn.eval()  #不启用 BatchNormalization 和 Dropout


##############################################
### Evaluate performance ###
##############################################

def evaluate(data_loader, mode):
    out = []
    xent = nn.CrossEntropyLoss()
    for data in data_loader:
        if type(data) == list:
            images, labels = data
        else:
            images = data

        images = Variable(images, requires_grad=True).cuda()
        images.retain_grad()    #添加扰动时需要用到梯度

        if mode == 'confidence':
            _, confidence = cnn(images)    #_,空出位置
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'confidence_scaling':
            epsilon = args.epsilon

            cnn.zero_grad()
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence).view(-1)
            loss = torch.mean(-torch.log(confidence))
            loss.backward()

            images = images - args.epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'baseline':
            pred, _ = cnn(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

        elif mode == 'ODIN':
            T = args.T
            epsilon = args.epsilon

            cnn.zero_grad()
            pred, _ = cnn(images)
            _, pred_idx = torch.max(pred.data, 1)
            labels = Variable(pred_idx)
            pred = pred / T
            loss = xent(pred, labels)   #ODIN使用交叉熵计算扰动，而confidence模式使用信心分数损失计算扰动
            loss.backward()

            images = images - epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            pred, _ = cnn(images)

            pred = pred / T
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

    out = np.concatenate(out)   
    return out


ind_scores = evaluate(ind_loader, args.process)
ind_labels = np.ones(ind_scores.shape[0])       #shape[0]取出维度的大小

ood_scores = evaluate(ood_loader, args.process)
ood_labels = np.zeros(ood_scores.shape[0])

labels = np.concatenate([ind_labels, ood_labels])
scores = np.concatenate([ind_scores, ood_scores])




#calculate detection errors
_,_,all_detection_errors, all_thresholds = detection(ind_scores, ood_scores)
np.save('plotdata\\all_detection_errors', all_detection_errors)
io.savemat('plotdata\\aall_detection_errors.mat', {'name': all_detection_errors})
np.save('plotdata\\all_thresholds', all_thresholds)
io.savemat('plotdata\\all_thresholds.mat', {'name': all_thresholds})



#calculate auroc
auroc = metrics.roc_auc_score(labels, scores)
np.save('plotdata\\auroc', auroc)
io.savemat('plotdata\\auroc.mat', {'name': auroc})

#calculate far
all_delta_far, all_far = far_threshhold(ind_scores, ood_scores)
np.save('plotdata\\all_delta_far', all_delta_far)
io.savemat('plotdata\\all_delta_far.mat', {'name': all_delta_far})
np.save('plotdata\\all_far', all_far)
io.savemat('plotdata\\all_far.mat', {'name': all_far})



#calculate mdr
all_delta_md, all_md = md_threshhold(ind_scores, ood_scores)
np.save('plotdata\\all_delta_md', all_delta_md)
io.savemat('plotdata\\all_delta_md.mat', {'name': all_delta_md})
np.save('plotdata\\all_md', all_md)
io.savemat('plotdata\\all_md.mat', {'name': all_md})


#calculate roc
all_delta_roc, all_far_roc, all_md_roc = roc(ind_scores, ood_scores)
np.save('plotdata\\all_far_roc', all_far_roc)
io.savemat('plotdata\\all_far_roc.mat', {'name': all_far_roc})
np.save('plotdata\\all_md_roc', all_md_roc)
io.savemat('plotdata\\all_md_roc.mat', {'name': all_md_roc})


#calculate Accuracy
acc,_,_,_,_ = acc(ind_loader, cnn)
np.save('plotdata\\acc60', acc)
io.savemat('plotdata\\acc60.mat', {'name': acc})





# confidense density 
ranges = (np.min(scores), np.max(scores))
plt.figure(figsize=(3.2, 2.5),dpi=600)
sns.distplot(ind_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Legitimate data packets')
sns.distplot(ood_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Illegitimate data packets')
plt.xlabel('Confidence',fontproperties = 'Times New Roman', fontsize=10)
plt.ylabel('Density',fontproperties = 'Times New Roman', fontsize=10)
#plt.title('gaosi',fontproperties = 'Times New Roman', fontsize=12,fontweight='bold')
plt.legend(prop={'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
})
plt.savefig('confidense-density.eps', format='eps', bbox_inches='tight')
plt.savefig('confidense-density.png', format='png', bbox_inches='tight')
plt.show()




#calculate confusion matrix and F1-score
y_test, y_pred = predict(ind_loader, cnn)
F1 = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
heatmap = sns.heatmap(cm, annot=True, fmt='.5g', cmap='Blues' )

plt.xlabel('Predicted label',fontproperties = 'Times New Roman', fontsize=10)
plt.ylabel('True label',fontproperties = 'Times New Roman', fontsize=10)

heatmap = plt.savefig('confusion matrix.png',dpi=800, edgecolor='r',transparent=False, bbox_inches=None)
heatmap = plt.savefig('confusion matrix.eps',dpi=800, edgecolor='r',transparent=False, bbox_inches=None)





















