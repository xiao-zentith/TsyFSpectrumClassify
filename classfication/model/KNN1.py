import os
import json
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


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


def load_data(data_list):
    """
    加载数据并返回特征和标签
    :param data_list: 包含文件路径和类别的列表
    :return: 特征数组, 标签数组, 标签映射字典
    """
    features = []
    labels = []
    label_map = {}

    for item in data_list:
        label = item['category']
        if label not in label_map:
            label_map[label] = len(label_map)

        file_path = item['input']
        x_coords, y_coords, matrix = read_matrix_from_file(file_path)

        # 展平矩阵为特征向量
        feature_vector = matrix.flatten()

        # 将横坐标和纵坐标展平并添加到特征向量中
        feature_vector = np.concatenate((x_coords, y_coords, feature_vector))

        features.append(feature_vector)
        labels.append(label_map[label])

    # 将列表转换为numpy数组
    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map


def main(json_config_path, num_runs=5):
    """
    主函数，加载数据、训练模型并评估性能
    :param json_config_path: JSON配置文件路径
    :param num_runs: 运行次数
    """
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    with open(json_config_path, 'r') as config_file:
        dataset_info = json.load(config_file)

    for run in range(num_runs):
        random_seed = 42 + run  # 使用不同的随机种子
        print(f"\nRun {run + 1}/{num_runs} with seed {random_seed}")

        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for fold_info in dataset_info:
            train_files = fold_info['train'] + fold_info['validation']  # 合并train和validation
            test_files = fold_info['test']

            X_train, y_train, label_map = load_data(train_files)
            X_test, y_test, _ = load_data(test_files)
            n_classes = len(label_map)

            # 将标签二值化
            y_train_bin = label_binarize(y_train, classes=list(range(n_classes)))
            y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

            # 数据归一化
            scaler = StandardScaler()
            X_train_scaled = X_train
            X_test_scaled = X_test

            # 使用GridSearchCV自动寻找最佳K值
            knn = KNeighborsClassifier()
            param_grid = {'n_neighbors': np.arange(1, 21)}
            grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train_bin)

            best_k = grid_search.best_params_['n_neighbors']
            print(f"Best k for Fold {fold_info['fold']}: {best_k}")

            # 使用最佳K值进行预测
            best_knn = KNeighborsClassifier(n_neighbors=best_k)
            classifier = OneVsRestClassifier(best_knn)
            y_scores = classifier.fit(X_train_scaled, y_train_bin).predict_proba(X_test_scaled)
            y_pred = classifier.predict(X_test_scaled)

            # 转换预测结果为单个类别标签
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_test_bin, axis=1)

            # 计算评估指标
            accuracy = accuracy_score(y_true_labels, y_pred_labels)
            precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
            recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            print(f"Fold {fold_info['fold']} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            # 打印混淆矩阵
            cm = confusion_matrix(y_true_labels, y_pred_labels)
            print(f"Confusion Matrix for Fold {fold_info['fold']}:")
            print(cm)

        # 输出当前运行的结果
        print(f"\nRun {run + 1} Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Run {run + 1} Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Run {run + 1} Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"Run {run + 1} Average F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        # 存储当前运行的结果以便后续计算总体平均值
        all_accuracies.extend(accuracies)
        all_precisions.extend(precisions)
        all_recalls.extend(recalls)
        all_f1s.extend(f1s)

    # 输出最终结果
    print(f"\nOverall Average Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Overall Average Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
    print(f"Overall Average Recall: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
    print(f"Overall Average F1 Score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")


json_config_path = r'../../dataset_classify/dataset_info.json'

# 调用主函数
main(json_config_path, num_runs=5)



