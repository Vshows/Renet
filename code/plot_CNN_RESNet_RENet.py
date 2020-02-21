import pickle as pkl
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import embed


filename = ['CNN']
color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
line_style = ['--', '-.', ':', '-']
marker = ['.', ',', 'o', 'v']





pkls = '../data/results/precision_recall_CNN.pkl'
outpacks = pkl.load(open(pkls, "rb"))
results = outpacks["result"]
allprob = []
allans = []
for i in range(len(results)):
    ans = [0] * 53
    ans[results[i][1]] = 1
    ans = ans[1:]
    #53长度的空离表
    allans.append(ans)
    #53长度的列表
    allprob.append(results[i][4])

allans = np.reshape(np.array(allans), (-1))
allprob = np.reshape(np.array(allprob), (-1))
curr_pkl_score = average_precision_score(allans, allprob)

precision, recall, _ = precision_recall_curve(allans, allprob)
plt.plot(recall, precision, color=color[1], linestyle=':', lw=1.8, label='CNN')

# ResCNN 9
# Fill in the path of the model that you wanna compare with

pkls = '../data/results/precision_recall_RENet.pkl'
outpacks = pkl.load(open(pkls, "rb"))
results = outpacks["result"]
allprob = []
allans = []
for i in range(len(results)):
    ans = [0] * 53
    ans[results[i][1]] = 1
    ans = ans[1:]
    allans.append(ans)
    allprob.append(results[i][4])

allans = np.reshape(np.array(allans), (-1))
allprob = np.reshape(np.array(allprob), (-1))
curr_pkl_score = average_precision_score(allans, allprob)
print(curr_pkl_score)
precision, recall, _ = precision_recall_curve(allans, allprob)
plt.plot(recall, precision, color=color[2], linestyle=':', lw=1.8, label='SE-ResCNN')

# my soa
#31
pkls = '../data/results/precision_recall_Resnet.pkl'
outpacks = pkl.load(open(pkls, "rb"))
results = outpacks["result"]
allprob = []
allans = []
for i in range(len(results)):
    ans = [0] * 53
    ans[results[i][1]] = 1
    ans = ans[1:]
    allans.append(ans)
    allprob.append(results[i][4])

allans = np.reshape(np.array(allans), (-1))
allprob = np.reshape(np.array(allprob), (-1))
curr_pkl_score = average_precision_score(allans, allprob)
print(curr_pkl_score)
precision, recall, _ = precision_recall_curve(allans, allprob)
plt.plot(recall, precision, color='cornflowerblue', linestyle='-', lw=2, label='ResNet')


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 0.6])
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('baselines_CNN_RESNet_RENet.png')

