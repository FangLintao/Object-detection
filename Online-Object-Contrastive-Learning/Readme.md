# Online Object Representations with Contrastive Learning [Currently working on]  
![image](https://github.com/FangLintao/Object-detection/blob/master/Online-Object-Contrastive-Learning/images/cover.png)  
## 1. Introduction  
Online Object Representations with Contrastive Learning, as self-supervised method, aims to detect objects and contrast objects' similarity on their appearance by online representation. The main reference paper is [" Online Object Representations with Contrastive Learning "](https://online-objects.github.io/)
## 2. Dataset
the recommended dataset is [Epic-Kitchen](https://epic-kitchens.github.io/2020-100), which contains 700 videos and 740GB in total size.
In this project, we consider a video with 200 secs as our data sources.  
Ⅰ. First 5,10,20,40,80,160 seconds as training dataset  
Ⅱ. The last 40 seconds, 160-200, as validation dataset  
the datafolder contains the following files  

    Ⅰ. Train.txt & Test.txt  
    Ⅱ. splited videos  
    Ⅲ. video frames  

### 2.1 Data Reading
##### For cutting off video into small videos  

    from utils.DataLoading import Video2Frame  
    vid2frm = Video2Frame("dataroot")  
    vid2frm.split_video("video_name", seconds=[(0,5),(0,10),(0,20),(0,40),(0,80),(0,160),(160,200)])  
    
##### For transfering video into frames  

    from utils.DataLoading import Video2Frame 
    vid2frm = Video2Frame("dataroot")   
    vid2frm.read_frame("video_name")

##### For reading video frames addresses  

    from utils.DataLoading import  Datareading  
    datareading = Datareading("./datasets")  
    trainset, valset = datareading.data_address_reading()

## 3. Online Contrastive Network
##### the main features:  
1. the longer video is, the more obvious objects' contrastion is  
2. self-supervised learning with human-made labels  
### 3.1 Archtecture
![image](https://github.com/FangLintao/Object-detection/blob/master/Online-Object-Contrastive-Learning/images/1.png)
###### Reference: "Online Object Representations with Contrastive Learning", section "3.3. Architecture and Embedding Space",Sören Pirk, Mohi Khansari, Yunfei Bai, Corey Lynch, Pierre Sermanet,Google Brain,[CS.CV] 10/Jun/2019  
##### Ⅰ. Faster-RCNN layer  
Aim: Crop objects from each frames  
Input: original frame size with 3 channels  
Processing: Faster-RCNN shouldn't be trained by video frames.

    from torchvision.models.detection import fasterrcnn_resnet50_fpn  
    model = fasterrcnn_resnet50_fpn(pretrained=True)  
    for param in model.parameters():  
        param.requires_grad = False

##### Ⅱ。 ResNet50 layer + additional 3 ResNet units   
Aim: Extract feature information from croped images  
Input: original size of cropped object images with 3 channel  
##### Ⅲ. OCN Embedding layer  
Aim: embedding features from resnet layers  
Input: [2048,1,1]  
Layer: one fully connected layer  
### 3.2 Loss
![image](https://github.com/FangLintao/Object-detection/blob/master/Online-Object-Contrastive-Learning/images/2.png)  
## Implementation
### Device Requirement
Ⅰ. GPU 8G  
Ⅱ. Pytorch

        OCN-training.py

## Reference
Ⅰ. "Online Object Representations with Contrastive Learning", section "3.3. Architecture and Embedding Space",Sören Pirk, Mohi Khansari, Yunfei Bai, Corey Lynch, Pierre Sermanet,Google Brain,[CS.CV] 10/Jun/2019  
