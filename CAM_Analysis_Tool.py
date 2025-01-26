import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from util.get_dataset import TrainDataset, TestDataset
import torch
from util.CNNmodel_CAM import *
import torch.nn.functional as F
import numpy as np
import random
from scipy.io import savemat 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setip_seed = 2023

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook
activations = {}

len_mark = 32
snr = 5
num_cls = 5
lambda_ACR = 0.001
lambda_SCR = 0.01
index_start = 0
index_end = 5000


model = ResNet20(num_cls).cuda()
modelweightfile = f'model/CGRL_{snr}dB_{len_mark}_{lambda_ACR}_{lambda_SCR}.pth'
model = torch.load(modelweightfile)
model.cuda()

#data and target
x, _, y, _ = TrainDataset('w', snr, len_mark)
data = torch.Tensor(x[index_start:index_end,:,:]).cuda()
target = torch.Tensor(y[index_start:index_end]).long().cuda()

#features from the layer before GAP
model.layer3.register_forward_hook(get_activation('layer3'))
model(data)
cam_output = activations['layer3']
cam = F.conv1d(cam_output, model.linear.weight.view(model.linear.out_features, cam_output.size(1), 1)) + model.linear.bias.unsqueeze(0).unsqueeze(2)

#get cam for corresponding class
cam_for_target  = torch.stack([cam[i, target[i], :] for i in range(target.size()[0])])

# normalize cam
min_vals = cam_for_target.min(dim=1, keepdim=True)[0]
cam_for_target  -= min_vals
max_vals = cam_for_target.max(dim=1, keepdim=True)[0]
normalized_cam_for_target = cam_for_target/max_vals

#interpolate cam to the size of signal sample. i.e, size = 1024
normalized_cam_for_target  = F.interpolate(normalized_cam_for_target.unsqueeze(1), size=data.size()[2], mode='linear', align_corners=False).squeeze(1)

normalized_cam_for_target = normalized_cam_for_target.detach().cpu().numpy()

data_dict = {'cam': normalized_cam_for_target}
savemat(f'save_CAM/Mask32_normalized_cam_for_target.mat', data_dict)

data_dict = {'data': data.detach().cpu().numpy()}
savemat(f'save_CAM/Mask32_data.mat', data_dict)

data_dict = {'target': target.detach().cpu().numpy()}
savemat(f'save_CAM/Mask32_target.mat', data_dict)