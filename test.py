import seaborn as sns #导入包
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from numpy import random
xtick=['0','1']
ytick=['0','1']
True_label = torch.tensor([0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1])
T_predict1 = torch.tensor([0,0,1,1,1,0,1,1,1,0,0,1,0,0,0,0])

True_label = True_label.numpy()
T_predict1 = T_predict1.numpy()

C1 = confusion_matrix(True_label, T_predict1) / len(True_label)

#print(C1)

sns.heatmap(C1,fmt='g',cmap="YlOrRd",annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick) #画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
#plt.show()

feats = random.rand(16,128)
tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(feats)
pos_index = (True_label == 1)
neg_index = (True_label == 0)
pos_tsne = tsne[pos_index]
neg_tsne = tsne[neg_index]

plt.figure(figsize=(8, 8))
plt.scatter(pos_tsne[:, 0], pos_tsne[:, 1], 1, color='#dcba58', label='rumors')
plt.scatter(neg_tsne[:, 0], neg_tsne[:, 1], 1, color='#ff6e5d', label='non-rumors')
plt.legend(loc='upper left')
plt.show()