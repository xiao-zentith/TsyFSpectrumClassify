import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_recall_curve, \
    roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns


class PLSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None

    def fit(self, X, y):
        self.pls.fit(X, y)
        self.classes_ = np.unique(y)  # 设置类别
        return self

    def predict(self, X):
        return np.argmax(self.pls.predict(X), axis=1)

    def decision_function(self, X):
        return self.pls.predict(X)

    def predict_proba(self, X):
        decisions = self.decision_function(X)
        probabilities = (decisions - decisions.min()) / (decisions.max() - decisions.min())
        return probabilities


def read_matrix_from_file(file_path):
    """
    从TXT文件中读取数据并构建矩阵
    :param file_path: 文件路径
    :return: 横坐标数组, 纵坐标数组, 数据矩阵
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取横坐标
    x_coords = list(map(float, lines[0].strip().split()))

    # 提取纵坐标和数据矩阵
    y_coords = []
    data_matrix = []
    for line in lines[1:]:
        parts = list(map(float, line.strip().split()))
        y_coords.append(parts[0])
        data_matrix.append(parts[1:])

    return np.array(x_coords), np.array(y_coords), np.array(data_matrix)


def load_data(input_folder):
    """
    加载所有数据并返回特征和标签
    :param input_folder: 输入文件夹路径
    :return: 特征数组, 标签数组, 标签映射字典
    """
    features = []
    labels = []
    label_map = {}

    for label_idx, label in enumerate(os.listdir(input_folder)):
        label_map[label] = label_idx
        label_folder = os.path.join(input_folder, label)

        if not os.path.isdir(label_folder):
            continue

        for txt_file in os.listdir(label_folder):
            if not txt_file.endswith('.txt'):
                continue

            file_path = os.path.join(label_folder, txt_file)
            x_coords, y_coords, matrix = read_matrix_from_file(file_path)

            # 展平矩阵为特征向量
            feature_vector = matrix.flatten()

            # 将横坐标和纵坐标展平并添加到特征向量中
            feature_vector = np.concatenate((x_coords, y_coords, feature_vector))

            features.append(feature_vector)
            labels.append(label_idx)

    # 将列表转换为numpy数组
    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map


def perform_feature_engineering(X):
    """
    进行特征工程操作，包括PCA降维
    :param X: 原始特征数组
    :return: 处理后的特征数组
    """
    print(f"Original shape of X: {X.shape}")

    # PCA降维
    pca = PCA(n_components=min(3, X.shape[1]))  # 保留最多3个特征
    X_pca = pca.fit_transform(X)

    print(f"Shape of X after PCA: {X_pca.shape}")
    return X_pca


def plot_classification_metrics(y_true, y_pred, y_scores, n_classes, label_map, classifier):
    """
    绘制分类指标图，包括P-R曲线、ROC曲线及其AUC值和混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param y_scores: 分类得分
    :param n_classes: 类别数量
    :param label_map: 标签映射字典
    :param classifier: 训练好的分类器
    """
    # 计算准确率、召回率、F1分数
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))

    # P-R曲线
    plt.figure(figsize=(24, 8))

    # ROC曲线
    plt.subplot(1, 4, 1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {list(label_map.keys())[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # P-R曲线
    plt.subplot(1, 4, 2)
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
        plt.plot(recall[i], precision[i], label=f'Class {list(label_map.keys())[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="best")

    # 混淆矩阵
    plt.subplot(1, 4, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # 特征重要性（示例）
    plt.subplot(1, 4, 4)
    try:
        # PLS-DA模型没有直接的特征重要性属性
        plt.text(0.5, 0.5, 'PLS-DA does not provide feature importance directly', horizontalalignment='center',
                 verticalalignment='center', fontsize=12)
    except AttributeError:
        plt.text(0.5, 0.5, 'Model does not provide feature importance', horizontalalignment='center',
                 verticalalignment='center', fontsize=12)

    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')

    plt.tight_layout()
    plt.show()


def main(train_folder, test_folder):
    """
    主函数，加载数据、训练模型并评估性能
    :param train_folder: 训练集文件夹路径
    :param test_folder: 测试集文件夹路径
    """
    # 加载训练集和测试集
    X_train, y_train, label_map = load_data(train_folder)
    X_test, y_test, _ = load_data(test_folder)
    n_classes = len(label_map)

    # 进行特征工程
    X_train = perform_feature_engineering(X_train)
    X_test = perform_feature_engineering(X_test)

    # 将标签二值化
    y_train_bin = label_binarize(y_train, classes=list(range(n_classes)))
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # 使用GridSearchCV自动寻找最佳参数
    pls_da = PLSClassifier()
    param_grid = {'n_components': np.arange(1, min(10, X_train.shape[1]) + 1)}
    grid_search = GridSearchCV(pls_da, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train_bin)

    best_n_components = grid_search.best_params_['n_components']
    print(f"Best number of components found: {best_n_components}")

    # 使用最佳参数进行预测
    best_pls_da = PLSClassifier(n_components=best_n_components)
    calibrated_classifier = CalibratedClassifierCV(best_pls_da, method='sigmoid', cv=5)
    classifier = OneVsRestClassifier(calibrated_classifier)
    classifier.fit(X_train, y_train_bin)

    y_scores = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)

    # 转换预测结果为单个类别标签
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test_bin, axis=1)

    # 绘制分类指标图
    plot_classification_metrics(y_true_labels, y_pred_labels, y_scores, n_classes, label_map, classifier)


# 设置输入文件夹路径
train_folder = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset\dataset_train'
test_folder = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset\dataset_test'

# 调用主函数
main(train_folder, test_folder)