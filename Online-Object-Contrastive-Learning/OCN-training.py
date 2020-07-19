#!/usr/bin/env python
# coding: utf-8

from utils.DataLoading import  Datareading
from utils.tensorboard_evaluation import Evaluation
from model.OCN import OCN
from model.Loss import Metric_Loss,Nearest_Neighbor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch

import os
import warnings
from tqdm import tqdm



Class = ["unlabeled","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",
"street sign","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
"giraffe","hat","backpack","umbrella","shoe","eye glasses","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife",
"spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
"bed","mirror","dining table","window","desk","toilet","door","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
"toaster","sink","refrigerator","blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush","hair brush"]


datareading = Datareading("./datasets")
transform = transforms.Compose([transforms.ToPILImage(), 
                                transforms.Resize((360,640)),
                                transforms.ToTensor()])
trainset, valset = datareading.data_address_reading()

def train(num_epoch, learning_rate = 2.5e-4, model_dir="./saved_models", tensorboard_dir="./tensorboard"):
    warnings.filterwarnings("ignore", category=FutureWarning)
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    # create tensrboard to record loss data
    tensorboard_Loss = Evaluation(tensorboard_dir,name = "Loss",stats=["val_Loss"])
    tensorboard_Training_Metric_Loss = Evaluation(tensorboard_dir,name = "Training_Metric_Loss",stats=["Train_Metric_Loss"])
    #tensorboard_Training_CE_Loss = Evaluation(tensorboard_dir,name = "Training_CE_Loss",stats=["Train_CE_Loss"])
    tensorboard_Training_Loss = Evaluation(tensorboard_dir,name = "Training_Loss",stats=["Train_Loss"])
    
    tensorboard_Val_Folder_Loss = Evaluation(tensorboard_dir,name = "Val_Folder_Loss",stats=["Val_Folder_Loss"])
    
    
    training_folder = trainset.keys()
    num_frame = [5,10,20,40,80,160]
    # iterate each folder to train certain epoches
    for batch in zip(training_folder,num_frame):
        folder,num = batch
        # reading training data
        train_data = datareading.data_reading(trainset[folder],transform)
        length = len(train_data)

        network = OCN(freeze_FasterRCNN = True,criterion=0.9).cuda()
        optimizer = optim.SGD( filter(lambda p: p.requires_grad, network.parameters()), lr=learning_rate ,momentum=0.9, weight_decay=1e-4 )
        Metric_LOSS = Metric_Loss()
        NNB = Nearest_Neighbor()
        CE_loss = nn.CrossEntropyLoss()
        LOSS = 0
        for epoch in tqdm(range(num_epoch),ascii=True, desc="epoch at {} ->>".format(folder)):
            train_loss = 0
            train_count = 0
            for anchor in range(length-1):
                anchor_image = train_data[anchor].cuda()
                optimizer.zero_grad()
                anchor_objects, anchor_features ,_ ,anchor_labels = network(anchor_image)
                ce_loss = CE_loss(anchor_objects, anchor_labels.cuda())
                #tensorboard_Training_CE_Loss.write_episode_data(anchor,{"Train_CE_Loss":ce_loss.cpu().detach()})
                del anchor_objects, anchor_labels
                me_count = 0
                me_loss = 0
                for item in range(anchor+1,length):
                    image_B = train_data[item].cuda()
                    _, features_B ,_ ,_ = network(image_B)
                    M_loss = Metric_LOSS.metric_loss(anchor_features,features_B)
                    #ce_loss = CE_loss(objects, labels.cuda())
                    #me_loss += (M_loss + ce_loss)
                    me_loss += M_loss
                    me_count += 1
                me_loss = me_loss/me_count + ce_loss
                tensorboard_Training_Metric_Loss.write_episode_data(epoch,{"Train_Metric_Loss":me_loss.detach().cpu()})
                me_loss.backward()
                optimizer.step()
                train_loss += me_loss.detach().cpu()
                train_count += 1
            train_loss = train_loss/train_count
            tensorboard_Training_Loss.write_episode_data(epoch,{"Train_Loss":train_loss})
            
            with torch.no_grad():
                val_data = datareading.data_reading(valset['Frame_video-(160, 200).mp4'],transform)
                length = len(val_data)
                val_loss = 0
                val_count = 0
                for anchor in range(length-1):
                    anchor_image = val_data[anchor].cuda()
                    objects, features ,_ ,labels = network(anchor_image)
                    ce_loss = CE_loss(objects, labels.cuda())
                    loss = NNB.mse_loss(features)
                    val_loss += (loss.cpu().detach()+ce_loss.cpu().detach())
                    val_count += 1
                val_loss = val_loss/val_count
                tensorboard_Val_Folder_Loss.write_episode_data(epoch,{"Val_Folder_Loss":val_loss})
                torch.save(network.state_dict() ,os.path.join(model_dir, "the_folder_{}_saved_model_at_epoch_{}_loss_{}.pth".format(folder,epoch,val_loss)))
        LOSS+=val_loss
        tensorboard_Loss.write_episode_data(num,{"Val_Loss":LOSS/num_epoch})



train(num_epoch = 20)

