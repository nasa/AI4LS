from monai.networks.nets import DenseNet121
import torch.nn as nn
import torch

class TransferLearningImageClassifier(nn.Module):
    """ Image Classification Model using MONAI DenseNet121 """

    def __init__(self, num_classes=1):
        super(TransferLearningImageClassifier, self).__init__()
        self.model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes)

    def forward(self, x):
        return self.model(x)
    
class CNN_Scratch(nn.Module):
    def __init__(self):
        super(CNN_Scratch, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsampling

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsampling

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Downsampling

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # Downsampling

        self.global_avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.conv4(x)))

        x = self.global_avg_pool(x) 
        x = torch.flatten(x, start_dim=1) 

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


