import os
import numpy as np
from sklearn.datasets import load_breast_cancer  # 示例数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns  # 用于更美观的混淆矩阵
import joblib
import time

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

# 2. 定义双重坐标下降法的线性SVM模型
class LinearSVM_DCD:
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4):
        self.C = C  # 正则化参数
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = 0.0
        self.alphas = None

    def fit(self, X, y):
        """
        使用双重坐标下降法训练线性SVM。
        X: numpy数组，形状为 (n_samples, n_features)
        y: numpy数组，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        self.alphas = np.zeros(n_samples)
        self.w = np.zeros(n_features)
        self.b = 0.0

        # 对y进行转换，使得标签为 {-1, 1}
        y = np.where(y == 0, -1, 1)

        for it in range(self.max_iter):
            alpha_prev = np.copy(self.alphas)
            for i in range(n_samples):
                # 计算样本i的决策函数
                decision = np.dot(self.w, X[i]) + self.b
                # 计算误差
                error = y[i] * decision
                if error < 1:
                    # 更新alpha_i
                    self.alphas[i] = min(max(self.alphas[i] + (1 - error) / (np.dot(X[i], X[i]) + 1e-12), 0), self.C)
                else:
                    self.alphas[i] = 0.0
                # 更新w和b
                self.w = np.dot(self.alphas * y, X)
                self.b = np.mean(y - np.dot(X, self.w))
            # 计算收敛条件
            diff = np.linalg.norm(self.alphas - alpha_prev)
            if diff < self.tol:
                print(f'Converged at iteration {it + 1}')
                break
        else:
            print('Reached maximum iterations without convergence')

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)

# 3. 训练与评估过程
def main():
    # 参数设置
    C = 1.0
    max_iter = 100000
    tol = 1e-4

    # 创建results目录
    results_dir = './results_DCD_GeneExpressionClassification'
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据
    print("Loading and preprocessing data...")
    X, y, class_names = load_gene_expression_data()

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集、验证集和测试集（60%训练，20%验证，20%测试）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 训练模型
    print("Training model using Dual Coordinate Descent...")
    svm = LinearSVM_DCD(C=C, max_iter=max_iter, tol=tol)
    start_time = time.time()
    svm.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 测试模型
    print("Evaluating on test data...")
    y_pred = svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print("Classification Report:")
    print(report)

    # 保存模型参数
    model_save_path = os.path.join(results_dir, 'svm_gene_expression_classification.npy')
    np.save(model_save_path, {'w': svm.w, 'b': svm.b, 'alphas': svm.alphas})
    print(f"Model parameters saved to {model_save_path}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_save_path, dpi=400)
    plt.close()
    print(f"Confusion matrix saved to {cm_save_path}")

    # 绘制ROC曲线
    # 对于二分类，计算正类的决策分数
    y_test_binary = np.where(y_test == 0, -1, 1)
    decision_scores = svm.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test_binary, decision_scores)
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
    roc_save_path = os.path.join(results_dir, 'roc_curve.png')
    plt.savefig(roc_save_path, dpi=400)
    plt.close()
    print(f"ROC curve saved to {roc_save_path}")

    # 可视化训练过程
    # 由于DCD训练过程不记录历史，这里创建空图作为占位
    plt.figure(figsize=(12, 5))
    
    # 损失曲线（不可用，绘制空图）
    plt.subplot(1, 2, 1)
    plt.plot([], [], label='Train Loss')
    plt.plot([], [], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # 准确率曲线（不可用，绘制空图）
    plt.subplot(1, 2, 2)
    plt.plot([], [], label='Train Accuracy')
    plt.plot([], [], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    training_curves_save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(training_curves_save_path, dpi=400)
    plt.close()
    print(f"Training curves saved to {training_curves_save_path}")

    # 绘制准确率条形图
    plt.figure(figsize=(8, 6))
    accuracies = [test_accuracy * 100]
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
