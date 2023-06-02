import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

arr = np.array([[0.449, 0.042, 0.447, 0.044, 0.430, 0.062], [0.081, 0.428, 0.087, 0.422, 0.121, 0.387]])
x_tick = ['0', '1', '0', '1', '0', '1']
y_tick = ['0', '1']
sns.heatmap(arr, fmt='g', cmap="RdBu_r", square=True, annot=True, cbar=True, xticklabels=x_tick, yticklabels=y_tick)
plt.title("Heatmap of confusion matrix",fontsize='large',fontweight='bold')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.show()

arr_hir = np.array([[0.827,0.847,0.812,0.829,0.807,0.843,0.825],[0.824,0.854,0.769,0.809,0.802,0.875,0.837],[0.846,0.809,0.857,0.832,0.879,0.837,0.858],[0.840,0.855,0.830,0.842,0.825,0.851,0.837],[0.814,0.823,0.799,0.812,0.897,0.851,0.873],[0.816,0.818,0.815,0.817,0.816,0.818,0.817],[0.772,0.854,0.656,0.742,0.720,0.889,0.795],[0.877,0.843,0.911,0.876,0.914,0.847,0.879]])
x_tick = ['ACC', 'P-F', 'R-F', 'F1-F', 'P-T', 'R-T','F1-T']
y_tick = ['EANN', 'MVAE', 'MCNN', 'CAFE', 'MKEMN', 'SAFE', 'att-RNN', 'RDNN']
sns.heatmap(arr_hir, fmt='g', cmap="RdBu_r", square=True, annot=True, cbar=True, xticklabels=x_tick, yticklabels=y_tick)
plt.title("Heatmap of cross-sectional comparison experiment",fontsize='large',fontweight='bold')
plt.xlabel('Evaluation values')
plt.ylabel('Methods')
plt.show()

arr_ab = np.array([[0.817,0.762,0.862,0.809,0.878,0.780,0.826],[0.869,0.829,0.906,0.858,0.910,0.837,0.872],[0.877,0.843,0.911,0.876,0.914,0.847,0.879]])
x_tick = ['ACC', 'P-F', 'R-F', 'F1-F', 'P-T', 'R-T','F1-T']
y_tick = ['Baseline', 'Baseline+CLIP', 'RDNN']
sns.heatmap(arr_ab, fmt='g', cmap="RdBu_r", square=True, annot=True, cbar=True, xticklabels=x_tick, yticklabels=y_tick)
plt.title("Heatmap of ablation experiments",fontsize='large',fontweight='bold')
plt.xlabel('Evaluation values')
plt.ylabel('Methods')
plt.show()