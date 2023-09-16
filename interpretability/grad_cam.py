# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:37

@author: mick.yi

"""
import numpy as np
import cv2
import torch


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index, device):

        self.net.zero_grad()
        inputs = inputs.to(device)
        output = self.net(inputs)  # [1,num_classes]

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot = np.ones((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        one_hot = one_hot.to(device)
        one_hot = torch.sum(one_hot * output)  # label只有当前label位置是概率值，其他位置都是0

        one_hot.backward(retain_graph=True)
        one_hot.detach_()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
        torch.cuda.empty_cache()
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        if np.max(cam) != 0:
            cam /= np.max(cam)
        # resize to 224*224
        del one_hot
        return cam


class GradCAM_1(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = []
        self.gradient = []
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        feature = output
        self.feature.append(feature.cpu().detach())
        # print("feature shape:{}".format(output.size())) #feature shape:torch.Size([1, 128, 6, 6])

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        # self.gradient = output_grad[0]
        grad = output_grad[0]
        self.gradient = [grad.cpu().detach()] + self.gradient

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
            
    def __call__(self, inputs, index, device):
 
        self.net.zero_grad()
        inputs = inputs.to(device)
        output = self.net(inputs)  # [1,num_classes]

        one_hot = torch.zeros(1,output.size()[-1])
        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)

        one_hot[0][index] = 1
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        one_hot = one_hot.to(device)
        one_hot = torch.sum(one_hot * output)  #label只有当前label位置是概率值，其他位置都是0

        one_hot.backward(retain_graph=True)

        gradient = self.gradient[0]
        gradient = torch.squeeze(gradient)
        weight = torch.mean(gradient,dim=[1,2])
        feature = self.feature[0]
        feature = torch.squeeze(feature)

        torch.cuda.empty_cache()
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        cam = feature * weight
        cam = torch.sum(cam,dim=0)
        cam = torch.nn.ReLU()(cam)

        cam -= torch.min(cam)
        if torch.max(cam) != 0:
            cam /= torch.max(cam)

        if torch.min(cam)<0:
            cam1 = cam.reshape(-1)
            t = []
            for i in cam1:
                e = i.item()
                if e< 0:
                    e = 0
                t.append(e)
            cam1 = torch.as_tensor(t)
            cam = cam1.reshape(cam.shape)
        del one_hot
        return cam
