from torch.autograd import Function
import numpy as np
from PIL import Image, ImageOps
import random
import torch
import torch.nn as nn
import torch.nn.functional as Fabs
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import f1_score

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class PlaceCrop(object):
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y
    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean  
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label_l, label_s, resize_size=64, crop_size=60,\
                 is_train = True):
        n = len(data)
        self.data = data
        self.label_l = label_l
        self.label_s = label_s
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.is_train = is_train        
    def __len__(self):
        return len(self.data)   
    # Get one sample
    def __getitem__(self, index):  
        labels_l = int(self.label_l[index])
        labels_s = int(self.label_s[index])
        img = self.data[index]
        if not self.is_train:
            img = img + np.random.randn(2,640) * 0.001
        else:
            img = img
        img = torch.tensor(img).float()
        return img, torch.tensor(labels_l).long(), torch.tensor(labels_s).long()

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()
        self.n_data = len(data_list)
        self.img_paths = []
        self.img_labels = []
        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])
    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths))
        imgs = imgs.convert('RGB')
        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)
        return imgs, labels
    def __len__(self):
        return self.n_data

def optimizer_scheduler(optimizer, p):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1.0 + 10  * p) * 0.75
    return optimizer
    
class SupDann(nn.Module):
    def __init__(self):
        super(SupDann, self).__init__()
        self.height = 77
        self.f = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=5),
                nn.BatchNorm1d(64),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(64, 50, kernel_size=5),
                nn.BatchNorm1d(50),
                nn.Dropout1d(),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(50, 50, kernel_size=3),
                nn.BatchNorm1d(50),
                nn.Dropout1d(),
                nn.MaxPool1d(2),
                nn.ReLU(),
            )
        self.lc = nn.Sequential(
            nn.Linear(50*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 3),
            nn.Sigmoid(),
        )
        self.fs = nn.Sequential(
            nn.Linear(50*self.height, 50*self.height),
            nn.BatchNorm1d(50*self.height),
            nn.ReLU(),
        )
        self.sc = nn.Sequential(
            nn.Linear(50*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 3),
            nn.Sigmoid(),
        )
        self.dc = nn.Sequential(
            nn.Linear(50*self.height, 2),
            nn.Sigmoid(),
        )
        self.dc1 = nn.Sequential(
            nn.Linear(50*self.height, 2),
            nn.Sigmoid(),
        )
    def forward(self, x,alpha):
        xl,xq = x
        latent_xl = self.f(xl)
        latent_xq = self.f(xq)
        latent_xl = latent_xl.view(-1, 50*self.height)
        latent_xq = latent_xq.view(-1, 50*self.height)
        x = self.lc(latent_xl)
        latent_xq_hier = self.fs(latent_xq)
        s = self.sc(latent_xq_hier)
        y_xl = GRL.apply(latent_xl, alpha)
        d_xl = self.dc(y_xl)
        y_xq = GRL.apply(latent_xq, alpha)
        d_xq = self.dc(y_xq)
        y_xq_hier = GRL.apply(latent_xq_hier, alpha)
        d_xq_hier = self.dc1(y_xq_hier)
        x=x.view(x.shape[0],-1)
        s=s.view(s.shape[0],-1)
        d_xl=d_xl.view(d_xl.shape[0],-1)
        d_xq=d_xq.view(d_xq.shape[0],-1)
        d_xq_hier=d_xq_hier.view(d_xq_hier.shape[0],-1)
        return x, (d_xl,d_xq,d_xq_hier), s,(latent_xl,latent_xq,latent_xq_hier)
        
class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

def test_st(epoch,report_ep,model,source_train,source_test,\
                    target_train,target_test,criterion_l,\
                    criterion_s,criterion_d):
    # only for target_test accuracy and loss function
    # Here we are looking for damage task to see if damage is predicted correctly or not
    # f1 score tells about detection of the damage
    device = 'cuda'
    losses_save = []
    accuracy_save = [] # 1st acc will be detection, 2nd acc will localization, 3rd acc will be quantification
    f1_save =[] # 1st f1 score will be detection,, # 2nd f1 score will be quantification
    model.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct_d = 0 # Number of correct damage detection
    correct_l = 0 # Number of correct damage location detection
    correct_s = 0 # Number of correct damage quantification detection
    pred_d = []  # detection as 0 or 1 for damage
    pred_l = []  # location detection for the damage
    pred_s = []  # quantification detection for the damage
    
    with torch.no_grad():
        for data, target_d, target_s in source_test:
            data, target_d, target_s = data.to(device), target_d.to(device), target_s.to(device)
            output, _, output_s, _ = model((data,data),0.)
            test_loss1 += float(criterion_l(output, target_d))  # sum up batch loss
            test_loss2 += float(criterion_s(output_s, target_s))  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_l = pred.cpu().numpy().squeeze()
            targets_d = target_d.cpu().numpy().squeeze()
            pred = output_s.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_s = pred.cpu().numpy().squeeze()
            targets_s=target_s.cpu().numpy().squeeze()
    idx_l = np.where(targets_d!=0)
    targets_l = targets_d[idx_l]
    pred = pred_l[idx_l]
    correct_l = float(np.sum(pred==targets_l))/len(pred)
    correct_s = float(np.sum(pred_s==targets_s))/len(pred_s)
    target = targets_d>0
    pred_d = pred_l>0
    correct_d = float(np.sum(pred_d==target))/len(pred_d)
    
    f1_save.append(f1_score(pred_d,target,average='macro'))
    f1_save.append(f1_score(pred_s,targets_s,average='macro'))
    test_loss1 /= len(target_test)
    test_loss2 /= len(target_test)
    accuracy_save.append(correct_d)
    accuracy_save.append(correct_l)
    accuracy_save.append(correct_s)
    if (epoch % report_ep==0):
        print('Epoch '+str(epoch))
        print('Source test set: Average loss: {:.4f}, {:.4f}'.format(
            test_loss1, test_loss2))
    model.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct_d = 0 # correct damage detection
    correct_l = 0 # correct damage location detection
    correct_s = 0 # correct damage quantification detection
    pred_d = []  # detection as 0 or 1 for damage
    pred_l = []  # location detection for the damage
    pred_s = []  # quantification detection for the damage
    with torch.no_grad():
        for data, target_d, target_s in target_test:
            data, target_d, target_s = data.to(device), target_d.to(device), target_s.to(device)
            output, _, output_s, _ = model((data,data),0.)
            test_loss1 += float(criterion_l(output, target_d))  # sum up batch loss
            test_loss2 += float(criterion_s(output_s, target_s))  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_l = pred.cpu().numpy().squeeze()
            targets_d = target_d.cpu().numpy().squeeze()
            pred = output_s.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_s = pred.cpu().numpy().squeeze()
            targets_s=target_s.cpu().numpy().squeeze()
    idx_l = np.where(targets_d!=0)
    targets_l = targets_d[idx_l]
    pred = pred_l[idx_l]
    correct_l = float(np.sum(pred==targets_l))/len(pred)
    correct_s = float(np.sum(pred_s==targets_s))/len(pred_s)
    target = targets_d>0
    pred_d = pred_l>0
    correct_d = float(np.sum(pred_d==target))/len(pred_d)
    
    f1_save.append(f1_score(pred_d,target,average='macro'))
    f1_save.append(f1_score(pred_s,targets_s,average='macro'))
    test_loss1 /= len(target_test)
    test_loss2 /= len(target_test)
    accuracy_save.append(correct_d)
    accuracy_save.append(correct_l)
    accuracy_save.append(correct_s)
    if (epoch % report_ep==0):
        print('Target test set: Average loss: {:.4f}, {:.4f}'.format(
            test_loss1, test_loss2))
    return losses_save,accuracy_save,f1_save
