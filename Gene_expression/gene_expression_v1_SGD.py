# V1
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.datasets import load_breast_cancer  # 示例数据集
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     precision_recall_fscore_support,
#     confusion_matrix,
#     roc_curve,
#     auc
# )
# import matplotlib.pyplot as plt
# import seaborn as sns  # 用于更美观的混淆矩阵
# import joblib
# import time
# from torch.utils.data import Dataset, DataLoader, TensorDataset

# # 1. 数据加载与预处理
# def load_gene_expression_data():
#     """
#     加载基因表达数据集。
#     此处以乳腺癌数据集为示例，实际使用中请替换为您的基因表达数据。
#     """
#     data = load_breast_cancer()
#     X = data.data  # 特征矩阵
#     y = data.target  # 标签
#     class_names = data.target_names
#     return X, y, class_names

# # 2. 创建PyTorch数据集
# class GeneExpressionDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.from_numpy(X).float()  # 转换为浮点张量
#         self.y = torch.from_numpy(y).long()   # 转换为长整型张量
    
#     def __len__(self):
#         return self.X.shape[0]
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # 3. 定义多类SVM模型
# class MultiClassSVM(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(MultiClassSVM, self).__init__()
#         self.linear = nn.Linear(input_dim, num_classes)
    
#     def forward(self, x):
#         return self.linear(x)

# # 4. 定义多类合页损失（Multi-class Hinge Loss）
# def multi_class_hinge_loss(outputs, labels, margin=1.0):
#     """
#     计算多类合页损失。
#     """
#     num_classes = outputs.size(1)
#     correct_class_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)  # [batch_size, 1]
#     margins = torch.clamp(outputs - correct_class_scores + margin, min=0)  # [batch_size, num_classes]
#     margins[torch.arange(outputs.size(0)), labels] = 0  # 不计算正确类的损失
#     loss = torch.mean(torch.sum(margins, dim=1))
#     return loss

# # 5. 训练函数
# def train_model(model, device, train_loader, optimizer, epoch, log_interval=100):
#     model.train()
#     epoch_loss = 0.0
#     correct = 0
#     total = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = multi_class_hinge_loss(outputs, target)
#         loss.backward()
#         optimizer.step()
        
#         epoch_loss += loss.item()
        
#         # 计算准确率
#         _, predicted = torch.max(outputs.data, 1)
#         correct += (predicted == target).sum().item()
#         total += target.size(0)
        
#         if (batch_idx + 1) % log_interval == 0:
#             print(f'Epoch [{epoch}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
#     avg_loss = epoch_loss / len(train_loader)
#     accuracy = correct / total
#     return avg_loss, accuracy

# # 6. 验证函数
# def validate_model(model, device, valid_loader):
#     model.eval()
#     valid_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, target in valid_loader:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             loss = multi_class_hinge_loss(outputs, target)
#             valid_loss += loss.item()
            
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == target).sum().item()
#             total += target.size(0)
    
#     avg_loss = valid_loss / len(valid_loader)
#     accuracy = correct / total
#     return avg_loss, accuracy

# # 7. 测试函数
# def test_model(model, device, test_loader, class_names):
#     model.eval()
#     test_loss = 0.0
#     correct = 0
#     total = 0
#     all_targets = []
#     all_predictions = []
#     all_scores = []
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             print("---------------------------------------------------------")
#             print(f"Output shape: {outputs.shape}")  # 添加此行以检查输出形状
#             print("---------------------------------------------------------")
#             loss = multi_class_hinge_loss(outputs, target)
#             test_loss += loss.item()
            
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == target).sum().item()
#             total += target.size(0)
            
#             all_targets.extend(target.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())
#             all_scores.extend(outputs.cpu().numpy())
    
#     avg_loss = test_loss / len(test_loader)
#     accuracy = correct / total
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         all_targets, all_predictions, average='weighted', zero_division=0
#     )
#     report = classification_report(all_targets, all_predictions, target_names=class_names, zero_division=0)
    
#     return avg_loss, accuracy, precision, recall, f1, report, all_targets, all_predictions, all_scores

# # 8. 绘制混淆矩阵
# def plot_confusion_matrix(targets, predictions, class_names, save_path):
#     cm = confusion_matrix(targets, predictions)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=400)
#     plt.close()
#     print(f"Confusion matrix saved to {save_path}")

# # 9. 绘制ROC曲线（每类一条曲线）
# def plot_roc_curve_multi_class(targets, scores, num_classes, class_names, save_path):
#     binary_targets = label_binarize(targets, classes=list(range(num_classes)))
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(num_classes):
#         fpr[i], tpr[i], _ = roc_curve(binary_targets[:, i], scores[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
    
#     # 计算micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(binary_targets.ravel(), scores.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
#     plt.figure(figsize=(12, 10))
#     for i in range(num_classes):
#         plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
#     plt.plot([0, 1], [0, 1], 'k--', lw=2)
#     plt.xlim([-0.01, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curves')
#     plt.legend(loc="lower right", fontsize='small', ncol=2)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=400)
#     plt.close()
#     print(f"ROC curves saved to {save_path}")



# # 10. 主函数
# def main():
#     # 参数设置
#     input_dim = 30  # 根据实际基因表达数据的特征数量调整
#     num_classes = 2  # 根据数据集的分类任务调整（例如，乳腺癌为2类）
#     learning_rate = 0.01
#     epochs = 50
#     batch_size = 16  # 样本数量较少，适当减小批量大小
#     momentum = 0.9
#     log_interval = 10  # 每10个批次打印一次日志
    
#     # 创建results目录
#     results_dir = './results_SGD_GeneExpressionClassification'
#     os.makedirs(results_dir, exist_ok=True)
    
#     # 检查设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # 加载数据
#     print("Loading and preprocessing data...")
#     X, y, class_names = load_gene_expression_data()
    
#     # 标准化特征
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
    
#     # 划分训练集、验证集和测试集（60%训练，20%验证，20%测试）
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X, y, test_size=0.4, random_state=42, stratify=y
#     )
#     X_valid, X_test, y_valid, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
#     )
    
#     # 创建PyTorch数据集
#     train_dataset = GeneExpressionDataset(X_train, y_train)
#     valid_dataset = GeneExpressionDataset(X_valid, y_valid)
#     test_dataset = GeneExpressionDataset(X_test, y_test)
    
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
#     # 初始化模型
#     model = MultiClassSVM(input_dim=input_dim, num_classes=num_classes)
#     model = model.to(device)
    
#     # 定义优化器
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
#     # 学习率调度器（可选）
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
#     # 记录训练过程
#     history = {
#         'train_loss': [],
#         'train_accuracy': [],
#         'valid_loss': [],
#         'valid_accuracy': []
#     }
    
#     # 训练与验证
#     for epoch in range(1, epochs + 1):
#         start_time = time.time()
#         train_loss, train_acc = train_model(model, device, train_loader, optimizer, epoch, log_interval)
#         valid_loss, valid_acc = validate_model(model, device, valid_loader)
#         end_time = time.time()
        
#         history['train_loss'].append(train_loss)
#         history['train_accuracy'].append(train_acc)
#         history['valid_loss'].append(valid_loss)
#         history['valid_accuracy'].append(valid_acc)
        
#         print(f'Epoch [{epoch}/{epochs}], '
#               f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, '
#               f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc * 100:.2f}%, '
#               f'Time: {end_time - start_time:.2f}s')
        
#         # 更新学习率
#         scheduler.step()
    
#     # 测试模型
#     print("Evaluating on test data...")
#     test_loss, test_acc, precision, recall, f1, report, test_targets, test_predictions, test_scores = test_model(
#         model, device, test_loader, class_names
#     )
#     print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%')
#     print("Classification Report:")
#     print(report)
    
#     # 保存模型
#     model_save_path = os.path.join(results_dir, 'svm_gene_expression_classification.pth')
#     torch.save(model.state_dict(), model_save_path)
#     print(f"Model saved to {model_save_path}")
    
#     # 绘制混淆矩阵
#     cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
#     plot_confusion_matrix(test_targets, test_predictions, class_names, cm_save_path)
    
#     # 绘制ROC曲线
#     roc_save_path = os.path.join(results_dir, 'roc_curve.png')
#     plot_roc_curve_multi_class(test_targets, np.array(test_scores), num_classes, class_names, roc_save_path)
    
#     # 可视化训练过程
#     epochs_range = range(1, epochs + 1)
    
#     plt.figure(figsize=(12, 5))
    
#     # 损失曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, history['train_loss'], label='Train Loss')
#     plt.plot(epochs_range, history['valid_loss'], label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
    
#     # 准确率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, [acc * 100 for acc in history['train_accuracy']], label='Train Accuracy')
#     plt.plot(epochs_range, [acc * 100 for acc in history['valid_accuracy']], label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Accuracy Curve')
#     plt.legend()
    
#     plt.tight_layout()
#     training_curves_save_path = os.path.join(results_dir, 'training_curves.png')
#     plt.savefig(training_curves_save_path, dpi=400)
#     plt.close()
#     print(f"Training curves saved to {training_curves_save_path}")
    
#     # 绘制准确率条形图（可选）
#     plt.figure(figsize=(8, 6))
#     accuracies = [test_acc * 100]
#     labels = ['Test Accuracy']
#     sns.barplot(x=labels, y=accuracies)
#     plt.ylim(0, 100)
#     plt.ylabel('Accuracy (%)')
#     plt.title('Model Accuracy')
#     accuracy_bar_save_path = os.path.join(results_dir, 'accuracy_bar.png')
#     plt.savefig(accuracy_bar_save_path, dpi=400)
#     plt.close()
#     print(f"Accuracy bar chart saved to {accuracy_bar_save_path}")
    
#     print(f"All visualizations have been saved to the {results_dir}/ directory.")

# if __name__ == "__main__":
#     main()

# V2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer  # 示例数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
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
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. 数据加载与预处理
def load_gene_expression_data():
    """
    加载基因表达数据集。
    此处以乳腺癌数据集为示例，实际使用中请替换为您的基因表达数据。
    """
    data = load_breast_cancer()
    X = data.data  # 特征矩阵
    y = data.target  # 标签
    class_names = data.target_names
    return X, y, class_names

# 2. 创建PyTorch数据集
class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # 转换为浮点张量
        self.y = torch.from_numpy(y).long()   # 转换为长整型张量
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. 定义多类SVM模型
class MultiClassSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassSVM, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# 4. 定义多类合页损失（Multi-class Hinge Loss）
def multi_class_hinge_loss(outputs, labels, margin=1.0):
    """
    计算多类合页损失。
    """
    num_classes = outputs.size(1)
    correct_class_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)  # [batch_size, 1]
    margins = torch.clamp(outputs - correct_class_scores + margin, min=0)  # [batch_size, num_classes]
    margins[torch.arange(outputs.size(0)), labels] = 0  # 不计算正确类的损失
    loss = torch.mean(torch.sum(margins, dim=1))
    return loss

# 5. 训练函数
def train_model(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = multi_class_hinge_loss(outputs, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch [{epoch}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 6. 验证函数
def validate_model(model, device, valid_loader):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = multi_class_hinge_loss(outputs, target)
            valid_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    
    avg_loss = valid_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 7. 测试函数
def test_model(model, device, test_loader, class_names):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_scores = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = multi_class_hinge_loss(outputs, target)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    report = classification_report(all_targets, all_predictions, target_names=class_names, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, report, all_targets, all_predictions, all_scores

# 8. 绘制混淆矩阵
def plot_confusion_matrix(targets, predictions, class_names, save_path):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# 9. 绘制ROC曲线（二分类）
def plot_roc_curve_binary(targets, scores, save_path):
    """
    绘制二分类的ROC曲线。
    """
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"ROC curve saved to {save_path}")

# 10. 主函数
def main():
    # 参数设置
    # 动态获取输入维度和类别数量
    X, y, class_names = load_gene_expression_data()
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    
    learning_rate = 0.01
    epochs = 300
    batch_size = 16  # 样本数量较少，适当减小批量大小
    momentum = 0.9
    log_interval = 10  # 每10个批次打印一次日志
    
    # 创建results目录
    results_dir = './results_SGD_GeneExpressionClassification'
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集、验证集和测试集（60%训练，20%验证，20%测试）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 创建PyTorch数据集
    train_dataset = GeneExpressionDataset(X_train, y_train)
    valid_dataset = GeneExpressionDataset(X_valid, y_valid)
    test_dataset = GeneExpressionDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 初始化模型
    model = MultiClassSVM(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)
    
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
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
        train_loss, train_acc = train_model(model, device, train_loader, optimizer, epoch, log_interval)
        valid_loss, valid_acc = validate_model(model, device, valid_loader)
        end_time = time.time()
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_acc)
        
        print(f'Epoch [{epoch}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc * 100:.2f}%, '
              f'Time: {end_time - start_time:.2f}s')
        
        # 更新学习率
        scheduler.step()
    
    # 测试模型
    print("Evaluating on test data...")
    test_loss, test_acc, precision, recall, f1, report, test_targets, test_predictions, test_scores = test_model(
        model, device, test_loader, class_names
    )
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%')
    print("Classification Report:")
    print(report)
    
    # 保存模型
    model_save_path = os.path.join(results_dir, 'svm_gene_expression_classification.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # 绘制混淆矩阵
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_targets, test_predictions, class_names, cm_save_path)
    
    # 绘制ROC曲线
    if num_classes == 2:
        # 对于二分类，使用单个分数
        # 通常使用正类的分数
        scores = np.array(test_scores)[:, 1]  # 选择第二列作为正类的分数
        plot_roc_curve_binary(test_targets, scores, os.path.join(results_dir, 'roc_curve.png'))
    else:
        # 多分类情况
        plot_roc_curve_multi_class(test_targets, np.array(test_scores), num_classes, class_names, os.path.join(results_dir, 'roc_curve.png'))
    
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
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history['train_accuracy']], label='Train Accuracy')
    plt.plot(epochs_range, [acc * 100 for acc in history['valid_accuracy']], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    training_curves_save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(training_curves_save_path, dpi=400)
    plt.close()
    print(f"Training curves saved to {training_curves_save_path}")
    
    # 绘制准确率条形图（可选）
    plt.figure(figsize=(8, 6))
    accuracies = [test_acc * 100]
    labels = ['Test Accuracy']
    sns.barplot(x=labels, y=accuracies)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    accuracy_bar_save_path = os.path.join(results_dir, 'accuracy_bar.png')
    plt.savefig(accuracy_bar_save_path, dpi=400)
    plt.close()
    print(f"Accuracy bar chart saved to {accuracy_bar_save_path}")
    
    print(f"All visualizations have been saved to the {results_dir}/ directory.")

if __name__ == "__main__":
    main()
