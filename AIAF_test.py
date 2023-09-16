import yaml
import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from utils.modelOurs import LoadModel, CNN
from utils.util import ae_loss, calc_gradient_penalty

import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import torch.utils.data as dataf
from scipy import io
from sklearn.decomposition import PCA  # 进行PCA降维
from torch.optim.lr_scheduler import MultiStepLR
from torchattacks.attacks.fgsm import FGSM
from torchattacks.attacks.pgd import PGD
from torchattacks.attacks.cw import CW
from heatmap import *

cudnn.benchmark = True
config_path = './config/adver.yaml'
conf = yaml.load(open(config_path,'r'), Loader=yaml.FullLoader)
exp_name = conf['exp_setting']['exp_name']
img_size = conf['exp_setting']['img_size']
img_depth = conf['exp_setting']['img_depth']

trainer_conf = conf['trainer']

if trainer_conf['save_checkpoint']:
    model_path = conf['exp_setting']['checkpoint_dir'] + exp_name+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = model_path+'{}'

# Random seed
np.random.seed(conf['exp_setting']['seed'])
_ = torch.manual_seed(conf['exp_setting']['seed'])

def loadData(args):
    if args.datasetName == 'Salinas':
        DataPath = 'dataset/Salinas/Salinas_corrected(1).mat'
        TRPath = 'dataset/Salinas/TRLabel.mat'
        TSPath = 'dataset/Salinas/TSLabel.mat'

        # load data
        Data = io.loadmat(DataPath)
        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)

        Data = Data['salinas_corrected']
        Data = Data.astype(np.float32)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']
    elif args.datasetName == 'HoustonU2018':
        DataPath = 'dataset/HoutonU2018/HoutonU2018_img.mat'
        TRPath = 'dataset/HoutonU2018/TRLabel.mat'
        TSPath = 'dataset/HoutonU2018/TSLabel.mat'

        # load data
        Data = io.loadmat(DataPath)
        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)

        Data = Data['HoutonU2018_img']
        Data = Data.astype(np.float32)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']
    else:
        DataPath = 'dataset/900(1000)_PaviaU03/paviaU.mat'
        TRPath = 'dataset/900(1000)_PaviaU03/TRLabel.mat'
        TSPath = 'dataset/900(1000)_PaviaU03/TSLabel.mat'

        # load data
        Data = io.loadmat(DataPath)
        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)

        Data = Data['paviaU']
        Data = Data.astype(np.float32)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']

    return Data, TrLabel, TsLabel

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def createPatches(X, y, windowSize, windowsizemax):
    [m, n, l] = np.shape(X)
    temp = X[:, :, 0]
    pad_width = np.floor(windowSize / 2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')
    for i in range(l):
        temp = X[:, :, i]
        pad_width = np.floor(windowSize / 2)
        pad_width = np.int_(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        # print("temp2:",temp2.shape)
        x2[:, :, i] = temp2

    [ind1, ind2] = np.where(y != 0)
    TrainNum = len(ind1)
    patchesData = np.empty((TrainNum, l, windowsizemax, windowsizemax), dtype='float32')
    patchesLabels = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    pad_width_enlarge = np.floor((windowsizemax - windowSize) / 2)
    pad_width_enlarge = np.int_(pad_width_enlarge)
    for i in range(len(ind1)):
        # patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patchh = np.empty((windowsizemax, windowsizemax, l), dtype='float32')
        for j in range(l):
            temp11 = patch[:, :, j]
            temp11 = np.pad(temp11, pad_width_enlarge, 'mean')
            patchh[:, :, j] = temp11

        patchh = np.reshape(patchh, (windowsizemax * windowsizemax, l))
        patchh = np.transpose(patchh)
        patchh = np.reshape(patchh, (l, windowsizemax, windowsizemax))
        patchesData[i, :, :, :] = patchh
        patchlabel = y[ind1[i], ind2[i]]
        patchesLabels[i] = patchlabel

    return patchesData, patchesLabels
def Normalize(dataset):
    [m, n, b] = np.shape(dataset)

    # change to [0,1]
    for i in range(b):
        _range = np.max(dataset[:, :, i]) - np.min(dataset[:, :, i])
        dataset[:, :, i] = (dataset[:, :, i] - np.min(dataset[:, :, i])) / _range

    return dataset
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
def TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize, windowSize_max):
    Data_TR, TR_gt_M = createPatches(Data, TR_gt, windowSize, windowSize_max)
    Data_TS, TS_gt_M = createPatches(Data, TS_gt, windowSize, windowSize_max)

    # change to the input type of PyTorch
    Data_TR = torch.from_numpy(Data_TR)
    Data_TS = torch.from_numpy(Data_TS)

    TrainLabel = torch.from_numpy(TR_gt_M) - 1
    TrainLabel = TrainLabel.long()
    TestLabel = torch.from_numpy(TS_gt_M) - 1
    TestLabel = TestLabel.long()

    return Data_TR, Data_TS, TrainLabel, TestLabel
def Full_test(args,Data,Label,trans_code,ae,targeted_model,attack=None,Classes=20):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    Part = 3
    part = args.batch_size
    pred_y = np.empty((len(Label)), dtype='float32')
    number = len(Label) // part
    test_num_correct_train = 0

    FGSM_attack = FGSM(targeted_model, eps=0.3,Classes=Classes)
    PGD_attack = PGD(targeted_model, eps=0.3, alpha=0.02, steps=20,Classes=Classes, random_start=True)
    CW_attack = CW(targeted_model, c=100, kappa=0, steps=20, lr=0.5)
    for i in range(number):
        ttempdata1 = Data[i * part:(i + 1) * part, :, :, :].to(device)
        TrainLabel_1 = Label[i * part:(i + 1) * part]
        TrainLabel_1 = TrainLabel_1.to(device)

        if attack == 'fgsm':
            ttempdata1 = FGSM_attack(ttempdata1,TrainLabel_1)
        elif attack == 'pgd':
            ttempdata1 = PGD_attack(ttempdata1,TrainLabel_1)
        elif attack == 'cw':
            ttempdata1 = CW_attack(ttempdata1,TrainLabel_1)
        else:
            ttempdata1 = ttempdata1

        ttempdata1 = (ttempdata1 * 2 - 1).to(device)
        fake_img = ae(ttempdata1,ttempdata1, insert_attrs=trans_code)[0]
        fake = (fake_img + 1) / 2
        out_C = torch.argmax(targeted_model(fake), 1)
        test_num_correct_train += torch.sum(out_C == TrainLabel_1, 0)

        ttemp_C = targeted_model(fake)
        ttemp_CC = torch.max(ttemp_C, 1)[1].squeeze()
        pred_y[i * part:(i + 1) * part] = ttemp_CC.cpu()

        del ttempdata1, ttemp_C, ttemp_CC, out_C

    if (i + 1) * part < len(Label):
        ttempdata1 = Data[(i + 1) * part:len(Label), :, :, :].to(device)
        TrainLabel_1 = Label[(i + 1) * part:len(Label)]
        TrainLabel_1 = TrainLabel_1.to(device)

        if attack == 'fgsm':
            ttempdata1 = FGSM_attack(ttempdata1,TrainLabel_1)
        elif attack == 'pgd':
            ttempdata1 = PGD_attack(ttempdata1,TrainLabel_1)
        elif attack == 'cw':
            ttempdata1 = CW_attack(ttempdata1,TrainLabel_1)
        else:
            ttempdata1 = ttempdata1

        ttempdata1 = (ttempdata1 * 2 - 1).to(device)
        fake_img = ae(ttempdata1, ttempdata1,insert_attrs=trans_code)[0]
        fake = (fake_img + 1) / 2

        out_C = torch.argmax(targeted_model(fake), 1)
        test_num_correct_train += torch.sum(out_C == TrainLabel_1, 0)

        ttemp_C = targeted_model(fake)
        ttemp_CC = torch.max(ttemp_C, 1)[1].squeeze()
        pred_y[(i + 1) * part:len(Label)] = ttemp_CC.cpu()

        del ttempdata1, ttemp_C, ttemp_CC, out_C
    accuracy = test_num_correct_train.item()/len(Label)
    return test_num_correct_train,accuracy
def main():
    parser = argparse.ArgumentParser(description='PyTorch')
    # 文中设置的参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help='# of images in each batch of data')
    parser.add_argument('--epochs', type=int, default=300,
                        help='# of epochs to train for')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='Initial learning rate value')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='value of weight dacay for regularization')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_path', default='./Train/checkpoint/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epsilon', type=float, default=0.3, metavar='LR',
                        help='adversarial rate (default: 0.3)')
    parser.add_argument('--datasetName', type=str, default='PaviaU',
                        help='PaviaU Salinas HoustonU2018')

    args = parser.parse_args()
    batch_size = args.batch_size
    torch.manual_seed(args.seed)

    print("CUDA Available: ", torch.cuda.is_available())
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    Data, TR_gt, TS_gt = loadData(args)
    [m, n, b] = np.shape(Data)

    Classes = len(np.unique(TR_gt)) - 1  # 除掉0
    # ===PCA====
    n_components = 3
    x = np.reshape(Data, (m * n, b))
    pca = PCA(n_components, copy=True, whiten=False)
    x = pca.fit_transform(x)
    _, b = x.shape
    Data = np.reshape(x, (m, n, b))
    print(Data.shape)

    Data = Normalize(Data)
    # ==================================================================================================================
    windowSize = 26
    img_size = windowSize
    if args.datasetName == 'Salinas':
        pretrained_model = "Train/ED_1_modelpath/Salinas_net_params_Nonormalize.pkl"
    elif args.datasetName == 'PaviaU':
        pretrained_model = "Train/ED_1_modelpath/1000_net_params_single4_Nonormalize.pkl"
    elif args.datasetName == 'HoustonU2018':
        pretrained_model = "Train/ED_1_modelpath/HoutonU2018_net_params_single4_Nonormalize.pkl"
    # targetModel, you can change it.
    targeted_model = CNN(n_components, Classes)
    targeted_model = targeted_model.to(device)
    targeted_model.load_state_dict(torch.load(pretrained_model))
    targeted_model.eval()
    print('Training window size:', windowSize)

    [Data_TR0, Data_TS0, TrainLabel, TestLabel] = TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize, windowSize)

    # Load Model
    ae_learning_rate = conf['model']['autoencoder']['lr']
    ae_betas = tuple(conf['model']['autoencoder']['betas'])
    dp_learning_rate = conf['model']['D_pix']['lr']
    dp_betas = tuple(conf['model']['D_pix']['betas'])

    ae = LoadModel('autoencoder', conf['model']['autoencoder'], img_size, img_depth, 1)
    d_pix = LoadModel('cnn', conf['model']['D_pix'], img_size, img_depth, None)
    # 共有特征部分的Special分支
    spec1 = LoadModel('autoencoder', conf['model']['autoencoder'], img_size, img_depth, 2)
    spec2 = LoadModel('autoencoder', conf['model']['autoencoder'], img_size, img_depth, 3)

    reconstruct_loss = torch.nn.MSELoss()
    # Use cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    ae = ae.cuda()
    ae = torch.nn.DataParallel(ae)
    d_pix = d_pix.cuda()
    d_pix = torch.nn.DataParallel(d_pix)

    spec1 = spec1.cuda()
    spec1 = torch.nn.DataParallel(spec1)
    spec2 = spec2.cuda()
    spec2 = torch.nn.DataParallel(spec2)
    trans_code = None

    # Loss weight setting
    loss_lambda = {}
    for k in trainer_conf['lambda'].keys():
        init = trainer_conf['lambda'][k]['init']
        loss_lambda[k] = init

    ae.load_state_dict(torch.load(args.save_path + 'PaviaU_AIAF.pkl'))
    # ==========================================Testing=============================================================================
    part = args.batch_size
    # 干净_训练 样本
    correct_num, accuracy = Full_test(args, Data_TR0, TrainLabel, trans_code, ae, targeted_model, attack=None, Classes=Classes)
    print('num_correct_train: ', correct_num.item(), ' |accuracy of clean train imgs in training set: %f\n' % accuracy)

    # 干净_测试 样本
    correct_num, accuracy = Full_test(args, Data_TS0, TestLabel, trans_code, ae, targeted_model, attack=None,Classes=Classes)
    print('num_correct_train: ', correct_num.item(), ' |accuracy of clean test imgs in testing set: %f\n' % accuracy)

    # FGSM 对抗样本
    correct_num, accuracy = Full_test(args, Data_TR0, TrainLabel, trans_code, ae, targeted_model, attack='fgsm',Classes=Classes)
    print('FGSM in training set|| num_correct_train: ', correct_num.item(), ' |accuracy : %f\n' % accuracy)

    correct_num, accuracy = Full_test(args, Data_TS0, TestLabel, trans_code, ae, targeted_model, attack='fgsm', Classes=Classes)
    print('FGSM in testing set|| num_correct_train: ', correct_num.item(), ' |accuracy : %f\n' % accuracy)
    # PGD 对抗样本
    correct_num, accuracy = Full_test(args, Data_TR0, TrainLabel, trans_code, ae, targeted_model, attack='pgd', Classes=Classes)
    print('PGD in training set|| num_correct_train: ', correct_num.item(), ' |accuracy : %f\n' % accuracy)

    correct_num, accuracy = Full_test(args, Data_TS0, TestLabel, trans_code, ae, targeted_model, attack='pgd',Classes=Classes)
    print('PGD in testing set|| num_correct_train: ', correct_num.item(), ' |accuracy : %f\n' % accuracy)

    # CW 对抗样本_训练
    correct_num, accuracy = Full_test(args, Data_TR0, TrainLabel, trans_code, ae, targeted_model, attack='cw', Classes=Classes)
    print('CW in training set|| num_correct_train: ', correct_num.item(), ' |accuracy : %f\n' % accuracy)

    correct_num, accuracy = Full_test(args, Data_TS0, TestLabel, trans_code, ae, targeted_model, attack='cw', Classes=Classes)
    print('CW in testing set|| num_correct_train: ', correct_num.item(), ' |accuracy : %f\n' % accuracy)

if __name__ == '__main__':
    main()
