import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import gc

# 1. 数据集获取
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])
    
    train_dataset = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../datasets', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

# 2. 数据预处理与特征扩展
def preprocess_and_expand_features(dataset, poly):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    expanded_features = []
    labels = []
    
    for batch_idx, (images, target) in enumerate(data_loader):
        # 展开图像
        images = images.view(images.size(0), -1).numpy()  # (batch_size, 784)
        
        # 添加多项式特征
        images_poly = poly.transform(images)  # 稀疏矩阵
        expanded_features.append(images_poly)
        labels.extend(target.numpy())
        
        print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")
        
    # 合并所有批次的数据
    X = np.vstack(expanded_features)
    y = np.array(labels)
    
    return X, y

# 3. 主函数
def main():
    # 加载数据
    train_dataset, test_dataset = load_mnist_data()
    
    # 定义多项式特征扩展（degree=1 保持原始特征）
    poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=False)
    
    print("Fitting PolynomialFeatures on training data...")
    # 仅拟合训练数据以避免数据泄漏
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
    all_train_images = []
    all_train_labels = []
    for images, target in train_loader:
        images = images.view(images.size(0), -1).numpy()
        all_train_images.append(images)
        all_train_labels.extend(target.numpy())
    X_train_raw = np.vstack(all_train_images)
    y_train = np.array(all_train_labels)
    
    # 拟合多项式特征
    poly.fit(X_train_raw)
    del all_train_images
    gc.collect()
    
    # 预处理训练数据
    print("Transforming training data with PolynomialFeatures...")
    X_train, y_train = preprocess_and_expand_features(train_dataset, poly)
    
    # 预处理测试数据
    print("Transforming testing data with PolynomialFeatures...")
    X_test, y_test = preprocess_and_expand_features(test_dataset, poly)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # 清理内存
    del poly
    gc.collect()
    
    # 使用SGDClassifier进行并行化训练
    print("Training SGDClassifier model...")
    # 使用管道结合标准化和SGDClassifier
    pipeline = make_pipeline(
        StandardScaler(with_mean=False),  # with_mean=False 适用于稀疏矩阵
        SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, n_jobs=-1, verbose=1)
    )
    
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 保存模型（可选）
    joblib.dump(pipeline, 'sgd_mnist_poly1.joblib')
    
    print("Model training completed.")
    
    # 预测
    print("Predicting on test data...")
    y_pred = pipeline.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    
    # 详细分类报告
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
