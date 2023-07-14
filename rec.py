
import torch
import torchvision
from PyQt5 import QtCore
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import sys
import PyQt5.QtGui as QtGui
from PyQt5.QtGui import QPainter, QPixmap, QPen, QScreen, QColor
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QPoint

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 创建一个程序界面，画图用的
class Winform(QWidget):
    def __init__(self, parent=None):
        super(Winform, self).__init__(parent)
        self.setWindowTitle("绘图例子")
        self.pix = QPixmap()  # 实例化一个 QPixmap 对象
        self.lastPoint = QPoint()  # 起始点
        self.endPoint = QPoint()  # 终点
        self.initUi()
        self.setFocus()

    def initUi(self):
        # 设置窗口大小
        self.resize(560, 560)
        # 设置画布大小，背景为白色
        self.pix = QPixmap(560, 560)
        self.pix.fill(Qt.black)

    # 重绘的复写函数 主要在这里绘制

    def paintEvent(self, event):
        pp = QPainter(self.pix)

        pen = QPen(Qt.white)  # 定义笔格式对象
        pen.setWidth(50)  # 设置笔的宽度
        pp.setPen(pen)  # 将笔格式赋值给 画笔

        # 根据鼠标指针前后两个位置绘制直线
        pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)  # 在画布上画出

    # 鼠标按压事件
    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint
        if event.button() == Qt.RightButton:
            self.initUi()

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == QtCore.Qt.Key_C:
            self.initUi()

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()
            # 截图窗口上绘制的图像
            Qimg = QScreen.grabWindow(app.primaryScreen(), self.winId()).toImage()
            for x in range(560):
                for y in range(560):
                    rgb = Qimg.pixel(y, x)
                    imgRGB[x][y][0] = QtGui.qRed(rgb)
                    imgRGB[x][y][1] = QtGui.qGreen(rgb)
                    imgRGB[x][y][2] = QtGui.qBlue(rgb)
            img = cv2.resize(imgRGB, (28, 28))
            predict(img)

# 定义一个一样的神经网络类
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 预测，根据你输入的图像预测出写的是哪个数字
def predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
    img = transform(img)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    # 扩展后，为[1，1，28，28]
    img = img.cuda()
    output = net(img)
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    # print(prob)  # prob是10个分类的概率
    pred = np.argmax(prob)  # 选出概率最大的一个
    print(pred.item())


if __name__ == '__main__':
    net = torch.load('LeNet5-6.pth')  # 加载模型
    net.eval()  # 把模型转为test模式

    # img = cv2.imread("../1.png")  # 读取要预测的图片

    imgRGB = np.zeros((560, 560, 3), dtype=np.float32)

    # 创建程序的界面
    app = QApplication(sys.argv)
    form = Winform()
    form.show()
    sys.exit(app.exec_())
