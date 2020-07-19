#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
class Metric_Loss:
    def __init__(self):
        pass
    def Distance(self,anchor_features, features):
        distance = torch.zeros(anchor_features.shape[0],features.shape[0])
        for anchor_idx, anchor_object in enumerate(anchor_features):
            for object_idx, Object in enumerate(features):
                D =torch.sqrt(torch.sum((anchor_object - Object)**2))
                distance[anchor_idx, object_idx] = D.item()
        return distance
    def pair_loss(self, anchor_image, ref_image):
        N = anchor_image.shape[0]
        ds = self.Distance(anchor_image,ref_image)
        _, index = torch.sort(ds)
        positive = [ref_image[i] for i in index[:,0]]
        positive_pair = [ torch.sum( anchor_image[i]*positive[i]) for i in range(N)] 
        loss = 0
        for i in range(len(positive_pair)):
            negative_pair = 0
            for negative in index[i,1:]:
                negative_pair += torch.exp(torch.sum(anchor_image[i]*ref_image[negative]) - positive_pair[i])
            loss += torch.log(1 + negative_pair)
        return 1/N * loss
    def metric_loss(self, image_A, image_B):
        return self.pair_loss(image_A,image_B) + self.pair_loss(image_B, image_A)

class Nearest_Neighbor:
    def __init__(self):
        pass
    def nearest_neighbor(self,features):
        Num = features.shape[0]
        distance = torch.ones(Num,Num)*100
        for i in range(Num):
            for j in range(Num):
                if i == j:
                    continue
                D =torch.sqrt(torch.sum((features[i] - features[j])**2))
                distance[i,j] = D.item()
        _, index = torch.sort(distance)
        NNB = []
        for idx, nearest_idx in enumerate(index[:,0]):
            NNB.append([(features[idx],features[nearest_idx]),(idx,nearest_idx)])
        return NNB
    def mse_loss(self, features):
        nnb = self.nearest_neighbor(features)
        number = len(nnb)
        MSE_loss = nn.MSELoss()
        LOSS = 0
        for i in nnb:
            Loss = MSE_loss(i[0][0], i[0][1])
            LOSS += Loss
        LOSS = LOSS / number
        return LOSS
