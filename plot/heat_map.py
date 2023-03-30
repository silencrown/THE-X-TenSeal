import matplotlib.pyplot as plt
import numpy as np

import torch

def save_heatmap(data, filename, xticks=None, yticks=None, show_number=False, X_label='X label', Y_label='Y label', cmap='Reds', fmt='.2f'):
    fig, ax = plt.subplots()
    
    im = ax.imshow(data, cmap=cmap)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置刻度标签
    if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks)
    if yticks is not None:
        ax.set_yticks(np.arange(len(yticks)))
        ax.set_yticklabels(yticks)

    # 在方块中添加数值标签
    if show_number:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, format(data[i, j], fmt), ha='center', va='center', color='w')

    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)

    # 保存图片
    plt.savefig(filename)

def attention_heapmap():
    attention_weights = torch.eye(10).reshape(( 10, 10))
    print(attention_weights)
    array = attention_weights.numpy()
    save_heatmap(array, './attention.png', X_label='Keys', Y_label='Queries')


if __name__ == '__main__':
    attention_heapmap()