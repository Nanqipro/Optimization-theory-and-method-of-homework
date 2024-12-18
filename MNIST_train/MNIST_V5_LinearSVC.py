import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.svm import LinearSVC
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
import joblib
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

# 2. 将PyTorch数据集转换为NumPy数组
def convert_to_numpy(dataset):
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images.view(images.size(0), -1).numpy()
    labels = labels.numpy()
    return images, labels

# 3. 绘制混淆矩阵
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

# 4. 绘制ROC曲线
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

# 5. 主函数
def main():
    # 参数设置
    C = 1.0  # 正则化参数
    max_iter = 1000000  # 最大迭代次数
    tol = 1e-4  # 容差
    
    # 创建results目录
    results_dir = './results_LinearSVC'
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查设备（虽然不使用GPU，但保留打印信息）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # 转换为NumPy数组
    X_train, y_train = convert_to_numpy(train_subset)
    X_valid, y_valid = convert_to_numpy(valid_subset)
    X_test, y_test = convert_to_numpy(test_dataset)
    
    # 将标签转换为-1和1
    y_train = np.where(y_train == 0, -1, 1)
    y_valid = np.where(y_valid == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)
    
    # 训练模型
    print("Training LinearSVC model using Dual Coordinate Descent (scikit-learn)...")
    start_time = time.time()
    svm_model = LinearSVC(C=C, max_iter=max_iter, tol=tol, dual=True, verbose=1)
    svm_model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training completed in {training_time:.2f} seconds.")
    
    # 保存模型（可选）
    joblib.dump(svm_model, os.path.join(results_dir, 'linear_svc_mnist.joblib'))
    print(f"Model saved to {os.path.join(results_dir, 'linear_svc_mnist.joblib')}")
    
    # 验证集评估
    y_valid_pred = svm_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
    
    # 测试集预测
    print("Predicting on test data...")
    y_test_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # 详细分类报告
    report = classification_report(y_test, y_test_pred, target_names=['-1', '1'])
    print("Classification Report:")
    print(report)
    
    # 计算决策函数分数，用于ROC曲线
    # 对于LinearSVC，可以使用 decision_function
    y_test_scores = svm_model.decision_function(X_test)
    
    # 绘制混淆矩阵
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_test_pred, cm_save_path)
    
    # 绘制ROC曲线
    roc_save_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curve(y_test, y_test_scores, roc_save_path)
    
    # 可视化训练过程（损失曲线和准确率曲线）
    # scikit-learn的LinearSVC不提供训练过程的详细信息，因此无法绘制收敛曲线。
    # 如果需要绘制收敛曲线，可以使用`verbose`参数的输出日志，或使用其他库如`liblinear`的详细日志。
    # 这里，我们仅绘制最终的准确率。
    
    # # 由于没有训练过程数据，创建一个简单的准确率条形图
    # plt.figure(figsize=(8, 6))
    # accuracies = [valid_accuracy * 100, test_accuracy * 100]
    # labels = ['Validation Accuracy', 'Test Accuracy']
    # sns.barplot(x=labels, y=accuracies)
    # plt.ylim(90, 100)
    # plt.ylabel('Accuracy (%)')
    # plt.title('Model Accuracy')
    # plt.savefig(os.path.join(results_dir, 'accuracy_bar.png'), dpi=400)
    # plt.close()
    # print(f"Accuracy bar chart saved to {os.path.join(results_dir, 'accuracy_bar.png')}")
    

    plt.figure(figsize=(8, 6))
    accuracies = [valid_accuracy * 100, test_accuracy * 100]
    labels = ['Validation Accuracy', 'Test Accuracy']

    # 创建 barplot 并设置颜色
    bar_colors = ['skyblue', 'salmon']  # 定义柱子的颜色
    ax = sns.barplot(x=labels, y=accuracies, palette=bar_colors)

    # 调整柱子的宽度
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - 0.5  # 设置新的宽度，这里是0.5

        # 我们改变每个柱子的位置和宽度
        patch.set_width(0.5)  # 设置新的宽度
        patch.set_x(patch.get_x() + diff * 0.5)  # 调整位置以保持中心不变

    plt.ylim(90, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')

    # 保存图像
    accuracy_bar_path = os.path.join(results_dir, 'accuracy_bar.png')
    plt.savefig(accuracy_bar_path, dpi=400, bbox_inches='tight')  # 使用 bbox_inches='tight' 防止裁剪
    plt.close()

    print(f"Accuracy bar chart saved to {accuracy_bar_path}")
    
    
    # 显示混淆矩阵和ROC曲线
    # 这里假设已经保存了混淆矩阵和ROC曲线的图片，您可以在本地查看
    print("All visualizations have been saved to the ./results_LinearSVC/ directory.")

if __name__ == "__main__":
    main()
