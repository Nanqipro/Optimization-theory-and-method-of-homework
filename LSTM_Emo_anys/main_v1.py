# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from sklearn.metrics import roc_curve, auc



# 超参数设置
data_path =  './data/data.txt'              # 数据集
vocab_path = './data/vocab.pkl'             # 词表
save_path = './saved_dict/lstm.ckpt'        # 模型训练结果
embedding_pretrained = \
    torch.tensor(
    np.load(
        './data/embedding_Tencent.npz')
    ["embeddings"].astype('float32'))
                                            # 预训练词向量
embed = embedding_pretrained.size(1)        # 词向量维度
dropout = 0.5                               # 随机丢弃
num_classes = 2                             # 类别数
num_epochs = 200                            # epoch数
batch_size = 128                            # mini-batch大小
pad_size = 50                               # 每句话处理成的长度(短填长切)
learning_rate = 1e-3                        # 学习率
hidden_size = 128                           # lstm隐藏层
num_layers = 2                              # lstm层数
MAX_VOCAB_SIZE = 10000                      # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号


def get_data():
    tokenizer = lambda x: [y for y in x]  # 字级别
    vocab = pkl.load(open(vocab_path, 'rb'))
    # print('tokenizer',tokenizer)
    print('vocab',vocab)
    print(f"Vocab size: {len(vocab)}")

    train,dev,test = load_dataset(data_path, pad_size, tokenizer, vocab)
    return vocab, train, dev, test

def load_dataset(path, pad_size, tokenizer, vocab):
    '''
    将路径文本文件分词并转为三元组返回
    :param path: 文件路径
    :param pad_size: 每个序列的大小
    :param tokenizer: 转为字级别
    :param vocab: 词向量模型
    :return: 二元组，含有字ID，标签
    '''
    contents = []
    n=0
    with open(path, 'r', encoding='gbk') as f:
        # tqdm可以看进度条
        for line in tqdm(f):
            # 默认删除字符串line中的空格、’\n’、't’等。
            lin = line.strip()
            if not lin:
                continue
            # print(lin)
            label,content = lin.split('	####	')
            # word_line存储每个字的id
            words_line = []
            # 分割器，分词每个字
            token = tokenizer(content)
            # print(token)
            # 字的长度
            seq_len = len(token)
            if pad_size:
                # 如果字长度小于指定长度，则填充，否则截断
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # 将每个字映射为ID
            # 如果在词表vocab中有word这个单词，那么就取出它的id；
            # 如果没有，就去除UNK（未知词）对应的id，其中UNK表示所有的未知词（out of vocab）都对应该id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            n+=1
            contents.append((words_line, int(label)))

    train, X_t = train_test_split(contents, test_size=0.4, random_state=42)
    dev,test= train_test_split(X_t, test_size=0.5, random_state=42)
    return train,dev,test
# get_data()

class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
    def __getitem__(self,index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label
    def __len__(self):
        return len(self.x)

# 以上是数据预处理的部分

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 定义LSTM模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 使用预训练的词向量模型，freeze=False 表示允许参数在训练中更新
        # 在NLP任务中，当我们搭建网络时，第一层往往是嵌入层，对于嵌入层有两种方式初始化embedding向量，
        # 一种是直接随机初始化，另一种是使用预训练好的词向量初始化。
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # bidirectional=True表示使用的是双向LSTM
        self.lstm = nn.LSTM(embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        # 因为是双向LSTM，所以层数为config.hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        # lstm 的input为[batchsize, max_length, embedding_size]，输出表示为 output,(h_n,c_n),
        # 保存了每个时间步的输出，如果想要获取最后一个时间步的输出，则可以这么获取：output_last = output[:,-1,:]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier
# xavier和kaiming是两种初始化参数的方法
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/acc.png', dpi=400)

def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('results/loss.png', dpi=400)
    
def plot_roc_curve(fpr, tpr, auc_score, save_path="results/roc_curve.png"):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=400)
    plt.close()



def train(model, dataloaders):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training...\n")
    plot_train_acc = []
    plot_train_loss = []

    for i in range(num_epochs):
        step = 0
        train_lossi = 0
        train_acci = 0
        for inputs, labels in dataloaders['train']:
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_lossi += loss.item()
            train_acci += metrics.accuracy_score(true, predic)

        dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), save_path)

        train_acc = train_acci / len(dataloaders['train'])
        train_loss = train_lossi / len(dataloaders['train'])
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)

        print(f"Epoch {i+1}/{num_epochs} - train_loss: {train_loss:.3f}, train_acc: {train_acc:.2f}, dev_loss: {dev_loss:.3f}, dev_acc: {dev_acc:.2f}")
    
    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)

    # Final Evaluation on Test Data
    # model.load_state_dict(torch.load(save_path))
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function, Result_test=True)
    print('================'*8)
    print(f"Test loss: {test_loss:.3f} | Test accuracy: {test_acc:.2%}")



def result_test(real, pred, save_folder="results"):
    # Calculate confusion matrix
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='micro')
    recall = recall_score(real, pred, average='micro')
    f1 = f1_score(real, pred, average='micro')
    print(f'test: acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')

    # Save Confusion Matrix
    labels = ['negative', 'positive']
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels)
    disp.plot(cmap="Blues", values_format='d')
    plt.savefig(f"{save_folder}/confusion_matrix.png", dpi=400)
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(real, pred)
    auc_score = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, auc_score, save_folder + "/roc_curve.png")
    
    # Compare Actual vs Predicted Values
    plt.figure(figsize=(8, 6))
    plt.plot(real[:100], label="Actual", color='blue')
    plt.plot(pred[:100], label="Predicted", color='red', linestyle='dashed')
    plt.xlabel('Sample Index')
    plt.ylabel('Class Label')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.savefig(f"{save_folder}/actual_vs_predicted.png", dpi=400)
    plt.close()


def plot_acc(train_acc, save_path="results/acc.png"):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    plt.plot(list(range(len(train_acc))), train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig(save_path, dpi=400)

def plot_loss(train_loss, save_path="results/loss.png"):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    plt.plot(list(range(len(train_loss))), train_loss, alpha=0.9, linewidth=2, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(save_path, dpi=400)

# 模型评估
def dev_eval(model, data, loss_function,Result_test=False):
    '''
    :param model: 模型
    :param data: 验证集集或者测试集的数据
    :param loss_function: 损失函数
    :return: 损失和准确率
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if Result_test:
        result_test(labels_all, predict_all)
    else:
        pass
    return acc, loss_total / len(data)

if __name__ == '__main__':
    # 设置随机数种子，保证每次运行结果一致，不至于不能复现模型
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = get_data()
    dataloaders = {
        'train': DataLoader(TextDataset(train_data), batch_size, shuffle=True),
        'dev': DataLoader(TextDataset(dev_data), batch_size, shuffle=True),
        'test': DataLoader(TextDataset(test_data), batch_size, shuffle=True)
    }
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model().to(device)
    init_network(model)
    train(model, dataloaders)
