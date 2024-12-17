import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 网络搭建
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    # 数据加载
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    # 实例化网络
    net = Net()
    net = net.to(device)

    print("initial accuracy:", evaluate(test_data, net))
    # 损失函数
    loss_fn = nn.NLLLoss()
    loss_fn = loss_fn.to(device)

    # 优化器
    learn_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    total_train_step = 0
    for epoch in range(5):
        for (x, y) in train_data:
            net.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = net.forward(x.view(-1, 28 * 28))

            loss = loss_fn(output, y)  # 对数损失函数

            loss.backward()

            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数: {},loss: {}".format(total_train_step, loss.item()))

        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

        torch.save(net, "my_net{}.pth".format(epoch))
        print("模型保存!")

    for (n, (x, _)) in enumerate(test_data):
        if n > 5:
            break
        x = x.to(device)  # Move data to GPU
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28).cpu().numpy())  # Move to CPU for visualization
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
