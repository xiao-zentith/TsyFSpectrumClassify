import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_recall_curve, \
    roc_curve, auc, confusion_matrix, precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from src.preprocessing.augment_data import AugmentData
from src.utils.data_io.matrix_reader import read_matrix_from_file
from src.utils.data_io.data_loader import load_data


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
    all_precisions = []
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
        precisions = []
        recalls = []
        f1s = []

        temp_folder = 'temp_augmented'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        test_norm_folder = 'temp_test'
        if not os.path.exists(test_norm_folder):
            os.makedirs(test_norm_folder)

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
            precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
            recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

            accuracies.append(accuracy)
            precisions.append(precision)
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

    #调用plot方法展示结果

# 设置输入文件夹路径
data_folder = r'C:\Users\xiao\Desktop\academic_papers\data\dataset_K\dataset_TsyF'

# 调用主函数
main(data_folder, num_runs=5)



