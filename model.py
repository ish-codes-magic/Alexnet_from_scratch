import torch
import torch.nn as nn

#Paper implementation: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
#the architecture consisis of 5 convolutional layers and 3 fully connected layers
#Convolutional layer 1: 96 filters of size 11x11 with stride 4
#Convolutional layer 2: 256 filters of size 5x5 with padding 2 and stride 1
#Convolutional layer 3: 384 filters of size 3x3 with padding 1 and stride 1
#Convolutional layer 4: 384 filters of size 3x3 with padding 1 and stride 1
#Convolutional layer 5: 256 filters of size 3x3 with padding 1 and stride 1
#We apply normalisation(Local Response Normalisation) after the first and second convolutional layers
#We apply max pooling after the first, second and fifth convolutional layers
#We use ReLU activation function after each convolutional layer

#The output of the last convolutional layer is fed to the fully connected layers
#The output of last convolutional layer is 256*6*6 = 9216 which is first flattened and then fed to the fully connected layers
#We initialise the network with a dropout of 0.5, pass it through 2 fully connected layers with 4096 neurons each and then finally to the output layer

class AlexNetModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,padding=2),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax(dim=1)
        )
        self.init_parameters()
        
    #initialise the weights and biases of the network according to the paper - Section 5
    def init_parameters(self):
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.constant_(self.convs[4].bias, 1)
        nn.init.constant_(self.convs[10].bias, 1)
        nn.init.constant_(self.convs[12].bias, 1)
        nn.init.constant_(self.classifier[2].bias, 1)
        nn.init.constant_(self.classifier[5].bias, 1)
        nn.init.constant_(self.classifier[7].bias, 1)
            
    def forward(self, x):
        x = self.convs(x)
        x = self.classifier(x)
        return x
    