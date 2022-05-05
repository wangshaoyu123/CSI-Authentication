import numpy as np
from models.vgg import VGG
import torch.nn.functional as F
from torch.autograd import Variable
import torch


def tpr95(ind_confidences, ood_confidences):
    #calculate the false positive error when tpr is 95%
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 100000

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):  #遍历所有的门限值
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))#fpr
        if tpr <= 0.94 and tpr >= 0.93:
            fpr += error2 #fpr对所有门限的累加值，后面再求均值
            total += 1

    fprBase = fpr / total  #求fpr均值

    return fprBase


def detection(ind_confidences, ood_confidences, n_iter=10000, return_data=True):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1)) #虚警率
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1)) #漏检率
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta
    # if return_data:
    #     return best_error, best_delta, all_errors, all_thresholds
    # else:
    #     return all_thresholds, all_errors
        #return best_error, best_delta
    
    
    
    
def far_threshhold(ind_confidences, ood_confidences):
    #calculate the falsepositive error
    Y1 = ood_confidences
    X1 = ind_confidences
    all_far = []
    all_delta = []
    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 10000
    for delta in np.arange(start, end, gap):  #遍历所有的门限值
        all_delta.append(delta)
        far = np.sum(np.sum(X1 < delta)) / np.float(len(X1)) #far
        all_far.append(far)
    return all_delta, all_far



def md_threshhold(ind_confidences, ood_confidences):
    #calculate the miss detection rate
    Y1 = ood_confidences
    X1 = ind_confidences
    all_md = []
    all_delta = []
    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 10000
    for delta in np.arange(start, end, gap):  #遍历所有的门限值
        all_delta.append(delta)
        md = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        all_md.append(md)
    return all_delta, all_md


def roc(ind_confidences, ood_confidences):
    #plot roc
    Y1 = ood_confidences
    X1 = ind_confidences
    all_far = []
    all_md = []
    all_delta = []
    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 10000
    for delta in np.arange(start, end, gap):  #遍历所有的门限值
        all_delta.append(delta)
        far = np.sum(np.sum(X1 < delta)) / np.float(len(X1))   #虚警率
        all_far.append(far) 
        md = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))   #漏检率
        all_md.append(md)
    return all_delta, all_far, all_md

def acc(loader, cnn):
    correct = []
    probability = []
    confidence = []

    for images, labels in loader:
        images = Variable(images).cuda()
        labels = labels.cuda()

        pred, conf = cnn(images)
        pred = F.softmax(pred, dim=-1)
        conf = F.sigmoid(conf).data.view(-1)
        
        pred_value, pred = torch.max(pred.data, 1)   #返回类别概率值和类别
        correct.extend((pred == labels).cpu().numpy())
        probability.extend(pred_value.cpu().numpy())
        confidence.extend(conf.cpu().numpy())

    correct = np.array(correct).astype(bool)
    probability = np.array(probability)
    confidence = np.array(confidence)
    
    #plot_histograms(correct, confidence)
    val_acc = np.mean(correct)    #准确率，妙啊
    conf_min = np.min(confidence)
    conf_max = np.max(confidence)
    conf_avg = np.mean(confidence)

    cnn.train()
    return val_acc, conf_min, conf_max, conf_avg, confidence




def predict(loader, cnn):
    pred_all = []
    label_all =[]
    for images, labels in loader:
        images = Variable(images).cuda()
        labels = labels.cuda()

        pred, conf = cnn(images)
        pred = F.softmax(pred, dim=-1)
        conf = F.sigmoid(conf).data.view(-1)
        
        pred_value, pred = torch.max(pred.data, 1)   #返回类别概率值和类别
        pred_all.extend(pred.cpu().numpy())
        label_all.extend(labels.cpu().numpy())

    return label_all, pred_all


