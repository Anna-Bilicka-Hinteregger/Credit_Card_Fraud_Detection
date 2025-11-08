import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📸 Confusion matrix saved to {save_path}")
    plt.show()


def plot_feature_importance(feature_importance, top_n=10, title="Top Features by Importance", save_path=None):
    feature_importance.head(top_n).plot(kind='barh', color='teal')
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Feature importance plot saved to {save_path}")
    plt.show()