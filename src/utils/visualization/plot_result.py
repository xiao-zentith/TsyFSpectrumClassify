from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, roc_curve, auc, \
    precision_recall_curve, confusion_matrix
import seaborn as sns


class plot_result:
    @staticmethod
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