import numpy as np
from data_process import get_MUSHROOM_data
import svm

# simple accuracy function
def get_acc(pred, y_test):
    return np.sum(y_test == pred) / len(y_test) * 100

# load dataset
# test is 0.2 by default (defined in get_MUSHROOM_data)
# we set validation to 0.2 of the remaining 0.8
VALIDATION = 0.2
data = get_MUSHROOM_data(VALIDATION)
X_train_MR, y_train_MR = data['X_train'], data['y_train']
X_val_MR, y_val_MR = data['X_val'], data['y_val']
X_test_MR, y_test_MR = data['X_test'], data['y_test']
n_class_MR = len(np.unique(y_test_MR))

print("Number of train samples: ", X_train_MR.shape[0])
print("Number of val samples: ", X_val_MR.shape[0])
print("Number of test samples: ", X_test_MR.shape[0])

# optimized hyperparameters
lr = 0.6
n_epochs = 10
reg_const = 0.04

# train the model
svm_MR = svm.SVM(n_class_MR, lr, n_epochs, reg_const)
svm_MR.train(X_train_MR, y_train_MR)

# get training accuracy
pred_svm = svm_MR.predict(X_train_MR)
print('The training accuracy is given by: %f' % (get_acc(pred_svm, y_train_MR)))

# get validation accuracy
pred_svm = svm_MR.predict(X_val_MR)
print('The validation accuracy is given by: %f' % (get_acc(pred_svm, y_val_MR)))

# get testing accuracy
pred_svm = svm_MR.predict(X_test_MR)
print('The testing accuracy is given by: %f' % (get_acc(pred_svm, y_test_MR)))