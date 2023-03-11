import seaborn as sns #导入包
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
xtick=['0','1']
ytick=['0','1']
True_label = torch.tensor([0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1])
T_predict1 = torch.tensor([0,0,1,1,1,0,1,1,1,0,0,1,0,0,0,0])

True_label = True_label.numpy()
T_predict1 = T_predict1.numpy()

C1 = confusion_matrix(True_label, T_predict1) / len(True_label)

print(C1)

sns.heatmap(C1,fmt='g',cmap="YlOrRd",annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick) #画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
plt.show()
