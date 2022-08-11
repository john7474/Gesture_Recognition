from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle

RANDOM_SEED = 42

dataset = 'landmark_data.csv'
model_save_path = 'gesture_recognition_model.hdf5'

NUM_CLASSES = 16

X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.8, random_state=0)

svm = SVC(C=10, gamma=0.1, kernel='rbf')
print(svm.fit(X_train, y_train))

y_pred = svm.predict(X_test)
print(y_pred)

cf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
print(f1, recall, precision)

with open('model.pkl','wb') as f:
    pickle.dump(svm,f)
