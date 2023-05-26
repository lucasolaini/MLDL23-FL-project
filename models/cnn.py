import torch.nn as nn

class CNN(nn.Module):

  def __init__(self, num_classes):
    super(CNN, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2))
    self.layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2))
    self.fc = nn.Sequential(
        # nn.Dropout(0.5),
        nn.Linear(64, 2048),
        nn.ReLU())
    self.fc1= nn.Sequential(
        nn.Linear(2048, num_classes))

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    out = self.fc1(out)
    return out