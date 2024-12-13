import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_recall_curve, \
    roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from Utils.augment_data import AugmentData


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
    :return: 特征数组, 标签数组, 标签映射字典, 文件路径列表
    """
    features = []
    labels = []
    label_map = {}
    file_paths = []

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
            file_paths.append(file_path)

    # 将列表转换为numpy数组
    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map, file_paths


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

    plt.tight_layout()
    plt.show()


def load_augmented_data(output_folder):
    """
    加载增强后的数据并返回特征和标签
    :param output_folder: 输出文件夹路径
    :return: 特征数组, 标签数组, 标签映射字典
    """
    features = []
    labels = []
    label_map = {}

    for label_idx, label in enumerate(os.listdir(output_folder)):
        label_map[label] = label_idx
        label_folder = os.path.join(output_folder, label)

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


def main(data_folder, num_runs=10):
    """
    主函数，加载数据、训练模型并评估性能
    :param data_folder: 数据集文件夹路径
    :param num_runs: 运行次数
    """
    all_accuracies = []
    all_recalls = []
    all_f1s = []

    for run in range(num_runs):
        random_seed = 42 + run  # 使用不同的随机种子
        print(f"\nRun {run + 1}/{num_runs} with seed {random_seed}")

        # 加载数据集
        X, y, label_map, file_paths = load_data(data_folder)
        n_classes = len(label_map)

        # 初始化StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        # 存储每个fold的结果
        accuracies = []
        recalls = []
        f1s = []

        temp_folder = 'temp_augmented'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        augmenter = AugmentData()

        for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
            print(f"Fold {fold_idx + 1}/{cv.get_n_splits()}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_files = [file_paths[idx] for idx in train_index]

            # 清空临时文件夹
            for root, dirs, files in os.walk(temp_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

            # 创建临时文件夹结构以存放增强数据
            for label in os.listdir(data_folder):
                label_folder = os.path.join(data_folder, label)
                temp_label_folder = os.path.join(temp_folder, label)

                if not os.path.isdir(label_folder):
                    continue

                if not os.path.exists(temp_label_folder):
                    os.makedirs(temp_label_folder)

            # 复制训练文件到临时文件夹
            for file_path in train_files:
                folder_name = os.path.basename(os.path.dirname(file_path))
                temp_subfolder = os.path.join(temp_folder, folder_name)
                output_file_path = os.path.join(temp_subfolder, os.path.basename(file_path))

                with open(file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
                    outfile.write(infile.read())

            # 对训练集进行数据增强
            augmenter.process_data(temp_folder, temp_folder, random_seed)

            # 加载增强后的训练数据
            X_train_augmented, y_train_augmented, _ = load_augmented_data(temp_folder)

            # 确保增强后的训练数据数量与预期一致
            assert len(X_train_augmented) == len(y_train_augmented), "Number of augmented training samples mismatch"

            # 将标签二值化
            y_train_bin = label_binarize(y_train_augmented, classes=list(range(n_classes)))
            y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

            # # 数据归一化
            scaler = StandardScaler()
            # X_train_augmented_scaled = scaler.fit_transform(X_train_augmented)
            # X_test_scaled = scaler.fit_transform(X_test)
            X_train_augmented_scaled = X_train_augmented
            X_test_scaled = X_test

            # 使用GridSearchCV自动寻找最佳K值
            knn = KNeighborsClassifier()
            param_grid = {'n_neighbors': np.arange(1, 21)}
            grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train_augmented_scaled, y_train_bin)

            best_k = grid_search.best_params_['n_neighbors']
            print(f"Best k for Fold {fold_idx + 1}: {best_k}")

            # 使用最佳K值进行预测
            best_knn = KNeighborsClassifier(n_neighbors=best_k)
            classifier = OneVsRestClassifier(best_knn)
            y_scores = classifier.fit(X_train_augmented_scaled, y_train_bin).predict_proba(X_test_scaled)
            y_pred = classifier.predict(X_test_scaled)

            # 转换预测结果为单个类别标签
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_test_bin, axis=1)

            # 计算评估指标
            accuracy = accuracy_score(y_true_labels, y_pred_labels)
            recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

            accuracies.append(accuracy)
            recalls.append(recall)
            f1s.append(f1)

            print(f"Fold {fold_idx + 1} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            # 打印混淆矩阵
            cm = confusion_matrix(y_true_labels, y_pred_labels)
            print(f"Confusion Matrix for Fold {fold_idx + 1}:")
            print(cm)

        # 清理临时文件夹
        for root, dirs, files in os.walk(temp_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_folder)

        # 输出当前运行的结果
        print(f"\nRun {run + 1} Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Run {run + 1} Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"Run {run + 1} Average F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        # 存储当前运行的结果以便后续计算总体平均值
        all_accuracies.extend(accuracies)
        all_recalls.extend(recalls)
        all_f1s.extend(f1s)

    # 输出最终结果
    print(f"\nOverall Average Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Overall Average Recall: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
    print(f"Overall Average F1 Score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")

# 设置输入文件夹路径
data_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\dataset_TsyF'

# 调用主函数
main(data_folder, num_runs=10)



