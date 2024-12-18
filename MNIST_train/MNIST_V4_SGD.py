import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns  # 用于更美观的混淆矩阵
import gc
import time

# 1. 数据集获取与预处理
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

# 2. 定义SVM模型
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输出维度为1，用于二分类
    
    def forward(self, x):
        return self.linear(x)

# 3. 定义Hinge Loss
def hinge_loss(outputs, labels, model=None, lambda_reg=0.01):
    # labels应为-1或1
    labels = labels.view(-1, 1).float()
    loss = torch.mean(torch.clamp(1 - labels * outputs, min=0))
    if model is not None:
        l2_reg = torch.tensor(0.).to(outputs.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += lambda_reg * l2_reg
    return loss

# 4. 训练函数
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 将标签转换为-1和1
        target = target.float()
        target = torch.where(target == 0, -1.0, target)  # 将标签0转换为-1
        target = torch.where(target > 0, 1.0, target)    # 其他标签保持为1
        
        optimizer.zero_grad()
        outputs = model(data.view(data.size(0), -1))
        loss = hinge_loss(outputs, target, model)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # 计算准确率
        predictions = torch.sign(outputs)
        predictions = predictions.view(-1).int()  # 转换为整数类型
        target_int = target.int()
        correct += (predictions == target_int).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 5. 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_scores = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 将标签转换为-1和1
            target = target.float()
            target = torch.where(target == 0, -1.0, target)
            target = torch.where(target > 0, 1.0, target)
            
            outputs = model(data.view(data.size(0), -1))
            loss = hinge_loss(outputs, target, model)
            test_loss += loss.item()
            
            predictions = torch.sign(outputs)
            predictions = predictions.view(-1).int()  # 转换为整数类型
            target_int = target.int()
            correct += (predictions == target_int).sum().item()
            total += target.size(0)
            
            all_targets.extend(target_int.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(outputs.view(-1).cpu().numpy())
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='binary', pos_label=1
    )
    report = classification_report(all_targets, all_predictions, target_names=['-1', '1'])
    
    return avg_loss, accuracy, precision, recall, f1, report, all_targets, all_predictions, all_scores

# 6. 绘制混淆矩阵
def plot_confusion_matrix(targets, predictions, save_path):
    cm = confusion_matrix(targets, predictions, labels=[-1, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['-1', '1'], yticklabels=['-1', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# 7. 绘制ROC曲线
def plot_roc_curve(targets, scores, save_path):
    # 将标签从-1,1转换为0,1
    binary_targets = np.where(np.array(targets) == 1, 1, 0)
    fpr, tpr, thresholds = roc_curve(binary_targets, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"ROC curve saved to {save_path}")

# 8. 主函数
def main():
    # 参数设置
    input_dim = 28 * 28  # MNIST图像尺寸
    learning_rate = 0.01
    epochs = 100
    batch_size = 64
    momentum = 0.9
    log_interval = 100  # 每100个批次打印一次日志
    lambda_reg = 0.01   # 正则化参数
    
    # 创建results目录
    results_dir = './results_SGD'
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 多GPU设置
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training.")
    
    # 加载数据
    train_dataset, test_dataset = load_mnist_data()
    
    # 划分部分训练集作为验证集（例如，使用10%的训练数据作为验证集）
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.seed(42)  # 为了可重复性
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    
    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)
    
    # 创建DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型
    model = SVM(input_dim)
    
    # 使用DataParallel进行多GPU训练
    if num_gpus > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'valid_loss': [],
        'valid_accuracy': []
    }
    
    # 训练与验证
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, log_interval)
        valid_loss, valid_acc, _, _, _, _, _, _, _ = test(model, device, valid_loader)
        end_time = time.time()
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_acc)
        
        print(f'\nEpoch: {epoch} \tTrain Loss: {train_loss:.6f} \tTrain Acc: {train_acc * 100:.2f}% '
              f'\tValid Loss: {valid_loss:.6f} \tValid Acc: {valid_acc * 100:.2f}% '
              f'\tTime: {end_time - start_time:.2f}s\n')
    
    # 测试模型
    test_loss, test_acc, precision, recall, f1, report, test_targets, test_predictions, test_scores = test(model, device, test_loader)
    print(f'Test Loss: {test_loss:.6f} \tTest Acc: {test_acc * 100:.2f}%')
    print("Classification Report:")
    print(report)
    
    # 保存混淆矩阵
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_targets, test_predictions, cm_save_path)
    
    # 保存ROC曲线
    roc_save_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curve(test_targets, test_scores, roc_save_path)
    
    # 可视化训练过程
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['valid_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    # plt.savefig(os.path.join(results_dir, 'loss_curve.png'), dpi=400)
    
    # plt.figure(figsize=(12, 5))
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history['train_accuracy']], label='Train Accuracy')
    plt.plot(epochs_range, [acc * 100 for acc in history['valid_accuracy']], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_accuracy_curve.png'), dpi=400)
    
    plt.tight_layout()
    # plt.show()
    
    # 保存模型
    # 如果使用DataParallel，需保存model.module.state_dict()
    model_save_path = 'svm_mnist.pth'
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# 运行主函数
if __name__ == "__main__":
    main()
