
#%%
#all imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
import numpy as np
import sys

Device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')






#%%
class VGGnet(nn.Module):    #what is that "Module" that i am calling to?
    """ this is the instance building a VGG16 (or other? ) CNN"""

    def __init__(self, num_of_classes):
        super(VGGnet, self).__init__()  #not sure what is that line exactly
        self.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1)
        self.conv11 = nn.Conv2d(64,64,3,stride=1,padding=1)
        
        self.conv2 = nn.Conv2d(64,128,3,stride=1,padding=1)
        self.conv22 = nn.Conv2d(128,128,3,stride=1,padding=1)

        self.conv3 = nn.Conv2d(128,256,3,stride=1,padding=1)
        self.conv33 = nn.Conv2d(256,256,3,stride=1,padding=1)

        self.conv4 = nn.Conv2d(256,512,3,stride=1,padding=1)
        self.conv44 = nn.Conv2d(512,512,3,stride=1,padding=1)  #used twice - before and after last maxpool
        
        self.fc1 = nn.Linear(512*7*7,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(num_of_classes,4096)
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = F.relu(self.conv1(x))   #3-64
        x = self.maxpool(F.relu(self.conv11(x))) #64-64
        x = F.relu(self.conv2(x))   #64-128
        x = self.maxpool(F.relu(self.conv22(x))) #128-128
        x = F.relu(self.conv3(x))   #128-256
        x = F.relu(self.conv33(x))   #256-256
        x = self.maxpool(F.relu(self.conv33(x))) #256-256
        x = F.relu(self.conv4(x))   #256-512
        x = F.relu(self.conv44(x))   #512-512
        x = self.maxpool(F.relu(self.conv44(x))) #512-512
        x = F.relu(self.conv44(x))   #512-512
        x = F.relu(self.conv44(x))   #512-512
        x = self.maxpool(F.relu(self.conv44(x))) #512-512

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

class VGGnet_smaller(nn.Module):    #what is that "Module" that i am calling to?
    """ this is the instance building a VGG16 (or other? ) CNN"""

    def __init__(self, num_of_classes):
        super(VGGnet_smaller, self).__init__()  #not sure what is that line exactly
        self.conv10 = nn.Conv2d(3,64,3,stride=1,padding=1)
        self.conv11 = nn.Conv2d(64,64,3,stride=1,padding=1)
        
        self.conv20 = nn.Conv2d(64,128,3,stride=1,padding=1)
        self.conv21 = nn.Conv2d(128,128,3,stride=1,padding=1)

        self.conv30 = nn.Conv2d(128,256,3,stride=1,padding=1)
        self.conv31 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.conv32 = nn.Conv2d(256,256,3,stride=1,padding=1)

        self.conv40 = nn.Conv2d(256,512,3,stride=1,padding=1)
        self.conv41 = nn.Conv2d(512,512,3,stride=1,padding=1)  #used twice - before and after last maxpool
        self.conv42 = nn.Conv2d(512,512,3,stride=1,padding=1)  #used twice - before and after last maxpool

        self.fc1 = nn.Linear(512*6*6,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_of_classes)
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = F.relu(self.conv10(x))   #3-64
        x = self.maxpool(F.relu(self.conv11(x))) #64-64
        
        x = F.relu(self.conv20(x))   #64-128
        x = self.maxpool(F.relu(self.conv21(x))) #128-128
        x = F.relu(self.conv30(x))   #128-256
        x = F.relu(self.conv31(x))   #256-256
        x = self.maxpool(F.relu(self.conv32(x))) #256-256
        x = F.relu(self.conv40(x))   #256-512
        x = F.relu(self.conv41(x))   #512-512
        x = self.maxpool(F.relu(self.conv42(x))) #512-512
        
        #x = x.view(-1, 512*6*6)
        batch_size = x.size()[0]
        #N,C,H,W = x.size() ## pytorch convevtion
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)

        return x

def make_VGGnet():
    net = VGGnet()
    return net

#%%
def train_net(epochs, net, loader, optimizer, loss_function=nn.CrossEntropyLoss()):
    """ a trainer function. 
        input:
        epochs;  net = the NN to train; 
        data - (input, labels)
        loss=nn.CrossEntropyLoss()
        optimizer - to be set outside the function and passed as a variable
        output:
        running_loss, net
    """
    running_loss = [] #a list to hold the loss for cheks
    loss_res = 1 # resolution of running_loss list
    for epoch in range(epochs):
        tmp_loss = 0.0  #for loss tracking every @loss_res steps
        for i, data in enumerate(loader,0):
            X, labels = data

            optimizer.zero_grad() #should i do that at the begginig or at the end?
            preds = net(X.to(Device))
            loss = loss_function(preds, labels.to(Device))
            loss.backward()
            optimizer.step()
            
            tmp_loss += loss.item()
            #printing loss to table
            if i % loss_res == 0:
                running_loss.append( (epoch + 1, i + 1, tmp_loss / loss_res) )
                tmp_loss = 0.0




            


    return net, running_loss        




## main

# necessry for downloading the CIFAR10 dataset - is there a nice way to encapsulate it in a class/other?
# instead of it being floating in the file?
# whats the "right" way to write in python - no need for "main" subrutine?
#%%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
trainset = torchvision.datasets.STL10(root='./data', split='train',
                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
testset = torchvision.datasets.STL10(root='./data', split='test',
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #for CIFAR10

classes = ('airplane', 'bird', 'car', 'cat', 
            'deer', 'dog', 'horse', 'monkey', 'ship', 'truck') #for STL10
## end of import data stuff
#%%
net  = VGGnet_smaller(len(classes))
net.to(device=Device)
lr = 0.001
optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
# XXXX = get_data()
net,running_loss = train_net(1, net, trainloader, optimizer)
####
y_test = testloader.dataset.data
pre_labels = net(y_test)

save_net(net)
save_results()

# %%
print(trainset)


# %%
for i, data in enumerate(trainloader,0):
    X, Y = data
    print(X[0,:])
    print(Y.tolist()[0])
    break
# %%


# %%
