
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 从网上下载数据集用于训练
train_set = datasets.MNIST('../dataset/',
                           train=True,
                           transform=transform,
                           download=True)
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True)

test_set = datasets.MNIST('../dataset/',
                          train=False,
                          transform=transform,
                          download=True)
test_loader = DataLoader(test_set,
                         batch_size=batch_size,
                         shuffle=False)


# 定义神经网络
class LeNet5(nn.Module):
    # 初始化神经网络结构
    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层1
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        # 卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        # 全连接层1
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        # 全连接层2
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # 全连接层3
        self.fc3 = nn.Linear(84, 10)

    # 计算反向传播（用于训练的函数）
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 训练
def train(epoch):
    net.train()  # 设置为training模式
    for batch_index, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device) # 将你的数据集移动到显卡中

        optimizer.zero_grad()

        outputs = net(data) # 将数据输入神经网络
        loss = criterion(outputs, target)  # 计算loss
        loss.backward()  # 反向传播
        optimizer.step()

        if batch_index % 300 == 299:  # 每 300 epoch 打印一次 loss 值
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))

# 计算loss值和精度
def test():
    net.eval()  # 设置为test模式
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = net(data)
            test_loss += F.cross_entropy(outputs, target, reduction='sum').item()  # sum up batch loss 把所有loss值进行累加
            pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))

    accuracy_list.append(100.0 * correct / len(test_loader.dataset))


epoch_list = []
accuracy_list = []

net = LeNet5() # 实例化这个神经网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 看你电脑有没有显卡
net.to(device) # 将神经网络移动到显卡或cpu上
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

if __name__ == '__main__':
    path = 'LeNet5-' # 模型保存路径
    for epoch in range(30): # 训练10轮
        train(epoch) # 训练
        test() # 计算loss
        epoch_list.append(epoch) # 将loss保存起来，用于训练完成后画图表

        name = path + str(epoch) + '.pth'
        torch.save(net, name)  # 保存模型

    plt.plot(epoch_list, accuracy_list) # 画图
    plt.ylabel('accuracy') # 纵坐标
    plt.xlabel('epoch') # 横坐标
    plt.show()