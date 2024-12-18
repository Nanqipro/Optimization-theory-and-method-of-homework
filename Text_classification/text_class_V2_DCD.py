import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns  # 用于更美观的混淆矩阵
import joblib
import time

# 1. 数据加载与预处理
def load_20newsgroups_data():
    # 加载训练和测试数据
    train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')  # 限制特征维度以控制计算量
    X_train = vectorizer.fit_transform(train_data.data)
    X_test = vectorizer.transform(test_data.data)
    
    # 获取标签
    y_train = train_data.target
    y_test = test_data.target
    class_names = train_data.target_names
    
    return X_train, y_train, X_test, y_test, class_names, vectorizer

# 2. 绘制混淆矩阵
def plot_confusion_matrix(targets, predictions, class_names, save_path):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# 3. 绘制ROC曲线（每类一条曲线）
def plot_roc_curve_multi_class(targets, scores, num_classes, class_names, save_path):
    binary_targets = label_binarize(targets, classes=list(range(num_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_targets[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_targets.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(12, 10))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"ROC curves saved to {save_path}")

# 4. 主函数
def main():
    # 参数设置
    input_dim = 20000  # TF-IDF特征维度
    num_classes = 20  # 20 Newsgroups
    C = 1.0  # 正则化参数
    max_iter = 1000000  # 最大迭代次数
    tol = 1e-4  # 容差
    
    # 创建results目录
    results_dir = './results_DCD_TextClassification'
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查设备（虽然不使用GPU，但保留打印信息）
    device = "CPU"  # scikit-learn的LinearSVC不支持GPU
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, class_names, vectorizer = load_20newsgroups_data()
    
    # 划分训练集和验证集（例如，80%训练，20%验证）
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 初始化模型
    svm_model = LinearSVC(C=C, max_iter=max_iter, tol=tol, dual=True, verbose=1, class_weight='balanced')
    
    # 训练模型
    print("Training LinearSVC model using Dual Coordinate Descent (scikit-learn)...")
    start_time = time.time()
    svm_model.fit(X_train_split, y_train_split)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training completed in {training_time:.2f} seconds.")
    
    # 保存模型
    model_save_path = os.path.join(results_dir, 'linear_svc_text_classification.joblib')
    joblib.dump(svm_model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # 验证集评估
    y_valid_pred = svm_model.predict(X_valid_split)
    valid_accuracy = accuracy_score(y_valid_split, y_valid_pred)
    print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
    
    # 测试集预测
    print("Predicting on test data...")
    y_test_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # 详细分类报告
    report = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # 计算决策函数分数，用于ROC曲线
    # 对于LinearSVC，可以使用 decision_function
    y_test_scores = svm_model.decision_function(X_test)
    
    # 绘制混淆矩阵
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_test_pred, class_names, cm_save_path)
    
    # 绘制ROC曲线
    roc_save_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curve_multi_class(y_test, y_test_scores, num_classes, class_names, roc_save_path)
    
    # 可视化训练过程
    # 由于LinearSVC不提供训练过程的详细信息，我们将使用验证集准确率作为示例
    # 这里，我们假设通过交叉验证或其他方法记录了每个epoch的验证准确率
    # 但scikit-learn的LinearSVC没有提供这种功能，因此无法绘制完整的收敛曲线
    
    # 为了展示，我们只绘制最终的验证准确率和测试准确率
    plt.figure(figsize=(8, 6))
    accuracies = [valid_accuracy * 100, test_accuracy * 100]
    labels = ['Validation Accuracy', 'Test Accuracy']
    sns.barplot(x=labels, y=accuracies)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_bar.png'), dpi=400)
    plt.close()
    print(f"Accuracy bar chart saved to {os.path.join(results_dir, 'accuracy_bar.png')}")
    
    print("All visualizations have been saved to the ./results_DCD_TextClassification/ directory.")

if __name__ == "__main__":
    main()
