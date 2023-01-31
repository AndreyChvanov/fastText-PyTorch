import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def conf_matrix(out, yb):
    count_cls = len(np.unique(yb))
    preds = torch.argmax(out, dim=1)
    m = confusion_matrix(y_true=yb, y_pred=preds)
    df_matrix = pd.DataFrame(m, range(count_cls), range(count_cls))
    sn.set(font_scale=1.4)
    sn.heatmap(df_matrix, annot=True, annot_kws={"size": 15}, fmt="d")
    plt.show()
