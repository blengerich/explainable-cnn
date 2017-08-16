
# coding: utf-8

# In[*]

import numpy as np
import matplotlib.pyplot as plt


svm_var = np.array([0.600, 0.6214, 0.6740, 0.720, 0.75, 0.75, 0.76, 0.75, 0.81, 0.79, 0.83, 0.85,
    0.89, 0.83, 0.73, 0.87, 0.85, 0.86, 0.83, 0.86, 0.82, 0.83, 0.89, 0.90, 0.84, 0.85, 0.89, 0.85,
    0.83, 0.84, 0.85, 0.80, 0.78, 0.81, 0.80, 0.79, 0.80, 0.81, 0.82, 0.79, 0.83, 0.81, 0.79, 0.79,
    0.84, 0.85, 0.84, 0.84, 0.84, 0.83, 0.85, 0.81, 0.84, 0.86, 0.83, 0.81, 0.82, 0.84, 0.85, 0.86, 0.85, 0.85])
svm_cor = np.array([0.488, 0.5214, 0.5122, 0.5212, 0.55, 0.65, 0.66, 0.65, 0.70, 0.72, 0.74, 0.76,
    0.71, 0.73, 0.62, 0.77, 0.65, 0.67, 0.79, 0.66, 0.62, 0.50, 0.63, 0.64, 0.54, 0.65, 0.69, 0.73,
    0.72, 0.71, 0.70, 0.72, 0.75, 0.79, 0.75, 0.77, 0.78, 0.71, 0.72, 0.69, 0.83, 0.71, 0.79, 0.79,
    0.79, 0.75, 0.74, 0.74, 0.69, 0.73, 0.80, 0.72, 0.79, 0.62, 0.81, 0.69, 0.74, 0.71, 0.72, 0.80, 0.81, 0.81])
val_acc = np.array([0.3521, 0.7174, 0.5702, 0.7802, 0.80, 0.79, 0.78, 0.82, 0.77, 0.75, 0.77, 0.82,
    0.73, 0.76, 0.78, 0.75, 0.82, 0.84, 0.83, 0.79, 0.82, 0.82, 0.42, 0.79, 0.83, 0.82, 0.81, 0.84,
    0.79, 0.70, 0.86, 0.85, 0.65, 0.86, 0.82, 0.80, 0.82, 0.80, 0.89, 0.82, 0.67, 0.86, 0.80, 0.86,
    0.89, 0.85, 0.84, 0.84, 0.89, 0.83, 0.90, 0.75, 0.90, 0.69, 0.85, 0.70, 0.76, 0.91, 0.90, 0.89, 0.90, 0.89])

fig = plt.figure()
#plt.subplot(111)
plt.plot(range(len(val_acc)), val_acc, 'b+')
plt.plot(range(len(svm_var)), svm_var, 'r.')
plt.plot(range(len(svm_cor)), svm_cor, 'gx')
plt.ylim([0, 1.0])
plt.xlim([-0.5, len(svm_cor)-0.5])
plt.title("Validation Set Accuracies during Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Full Network", "Activation Precision",
            "Activation-Output Correlation"], loc="best")
plt.savefig("svm_accuracy.png")
plt.show()


similarity = np.array([0.16, 0.20, 0.20, 0.24, 0.25, 0.25, 0.26, 0.29, 0.27, 0.28, 0.30, 0.32,
                       0.33, 0.35, 0.33, 0.37, 0.35, 0.38, 0.39, 0.40, 0.36, 0.35, 0.42, 0.44, 0.35, 0.36, 0.36, 0.36,
                       0.37, 0.38, 0.38, 0.37, 0.37, 0.40, 0.40, 0.39, 0.40, 0.41, 0.42, 0.38, 0.41, 0.37, 0.39, 0.39,
                       0.39, 0.37, 0.40, 0.41, 0.42, 0.43, 0.44, 0.41, 0.45, 0.42, 0.43, 0.43, 0.43, 0.44, 0.42, 0.42, 0.41, 0.40])

fig = plt.figure()
#plt.subplot(111)
plt.plot(range(len(val_acc)), val_acc, 'b+')
plt.plot(range(len(similarity)), similarity, 'r.')
plt.ylim([0, 1.0])
plt.xlim([-0.5, len(similarity)-0.5])
plt.title("Similarity of Sets of Selected Neurons during Training")
plt.xlabel("Epoch")
plt.ylabel("Set Similarity")
plt.legend(["Full Network Accuracy",
            "Selected Set Similarity"], loc="lower right")
plt.savefig("set_similarity.png")
plt.show()


# In[*]



