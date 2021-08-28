from sklearn.linear_model import RidgeClassifierCV
from pyts.datasets import load_basic_motions
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from pyts.multivariate.classification import MultivariateClassifier
from pyts.classification import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.interval_based import RandomIntervalSpectralForest
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.panel.reduce import Tabularizer
from pickle5 import pickle
import joblib
import struct
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from pyts.datasets import load_gunpoint
from sklearn.metrics import confusion_matrix

from sktime.classification.shapelet_based import ShapeletTransformClassifier

"""******"""
from sktime.classification.dictionary_based import BOSSEnsemble

from sktime.classification.dictionary_based import WEASEL

"""******"""


# try with univariate & with multivariate


print(struct.calcsize("P")*8)
# da = pd.read_csv("SisFall_dataset/SA01/D01_SA01_R01.txt" , delimiter=',' , header = None)
s = "SisFall_dataset/"

print("sup")
# # print(df.head)

# da[8] = da[8].replace({';':''} , regex = True)
# print(df)

# print(int( da.iloc[ : ,  8].values))

# ignore Y axis as will create false positives
"""
feature columns :
- XAD
- YAD
- ZAD
- XR
- YR
- ZR 
- XM
- YM
- ZM 

"""
arr = ["XAD", "YAD", "ZAD", "XR", "YR", "ZR", "XM", "YM", "ZM"]

# actual features selected : can discard the y-axis data because sensor was placed in line with centre of mass
chosen = ["XAD", "ZAD", "XR", "ZR"]

da = defaultdict(list)
de = defaultdict(list)
# print(df.loc[8])
#SA01, DA01_SAO1_R01
ya = []

ye = []
rec = defaultdict(list)

co = 0

f = open("tsfNoE=30.txt", 'w')


def TimeSeriesF(X_train, y_train, X_test, y_test):

    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=30))
    ]
    clf = Pipeline(steps, verbose=1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=['F', 'D']).ravel()

    arr = []
    arr.append(tn)
    arr.append(fp)
    arr.append(fn)
    arr.append(tp)
    score = (tn + tp) / (tn + fp + fn + tp)

    f.write(str(arr) + "   " + str(score) + "\n")

    # print("Feature : {} ".format(chosen[i]))
    print(score)


def rise(X_train, y_train, X_test, y_test):
    global f
    # univariate time series classification
    # train on extracted features
    for i in range(4):
        model = RandomIntervalSpectralForest(n_estimators=50)

        model.fit(X_train[[i]], y_train)

        y_pred = model.predict(X_test[[i]])

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=['F', 'D']).ravel()

        arr = []
        arr.append(tn)
        arr.append(fp)
        arr.append(fn)
        arr.append(tp)
        score = (tn + tp) / (tn + fp + fn + tp)

        f.write(chosen[i] + ": " + str(arr) + "   " + str(score) + "\n")

        print("Feature : {} ".format(chosen[i]))
        print(score)


def shapelet_transform(X_train, y_train, X_test, y_test):

    global f

    for i in range(4):
        model = ShapeletTransformClassifier(
            time_contract_in_mins=0.5, n_estimators=5)

        model.fit(X_train[[i]], y_train)

        print("yeee")
        y_pred = model.predict(X_test[[i]])

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=['F', 'D']).ravel()

        arr = []
        arr.append(tn)
        arr.append(fp)
        arr.append(fn)
        arr.append(tp)
        score = (tn + tp) / (tn + fp + fn + tp)

        f.write(chosen[i] + ": " + str(arr) + "   " + str(score) + "\n")

        print("Feature : {} ".format(chosen[i]))
        print(score)


def boss(X_train, y_train, X_test, y_test):
    global f

    for i in range(4):
        model = BOSSEnsemble(threshold=0.8, max_ensemble_size=100)

        model.fit(X_train[[i]], y_train)

        print("yeee")
        y_pred = model.predict(X_test[[i]])

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=['F', 'D']).ravel()

        arr = []
        arr.append(tn)
        arr.append(fp)
        arr.append(fn)
        arr.append(tp)
        score = (tn + tp) / (tn + fp + fn + tp)

        f.write(chosen[i] + ": " + str(arr) + "   " + str(score) + "\n")
        print(score)


def wesal(X_train, y_train, X_test, y_test):
    global f

    for i in range(4):
        model = WEASEL(bigrams=False, window_inc=5)

        model.fit(X_train[[i]], y_train)

        print("yeee")
        y_pred = model.predict(X_test[[i]])

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=['F', 'D']).ravel()

        arr = []
        arr.append(tn)
        arr.append(fp)
        arr.append(fn)
        arr.append(tp)
        score = (tn + tp) / (tn + fp + fn + tp)

        f.write(chosen[i] + ": " + str(arr) + "   " + str(score) + "\n")
        print(score)


# def uni_rocket(X_train, y_train, X_test, y_test):
#     global f

#     for i in range(4):
#         rocket = Rocket()

#         rocket.fit(X_train[[i]])
#         X_train_transform = rocket.transform(X_train[[i]])

#         model = RidgeClassifierCV()
#         model.fit(X_train_transform, y_train)

#         X_test_transform = rocket.transform(X_test[[i]])

#         y_pred = model.predict(X_test_transform)

#         tn, fp, fn, tp = confusion_matrix(
#             y_test, y_pred, labels=['F', 'D']).ravel()

#         arr = []
#         arr.append(tn)
#         arr.append(fp)
#         arr.append(fn)
#         arr.append(tp)
#         score = (tn + tp) / (tn + fp + fn + tp)

#         f.write(chosen[i] + ": " + str(arr) + "   " + str(score) + "\n")

#         print("Feature : {} ".format(chosen[i]))
#         print(score)


# def multi_rocket(X_train, y_train, X_test, y_test):

#     global f

#     rocket = Rocket()

#     rocket.fit(X_train)

#     X_train_transform = rocket.transform(X_train)

#     model = RidgeClassifierCV()

#     model.fit(X_train_transform, y_train)

#     X_test_transform = rocket.transform(X_test)

#     y_pred = model.predict(X_test_transform)

#     tn, fp, fn, tp = confusion_matrix(
#         y_test, y_pred, labels=['F', 'D']).ravel()

#     arr = []
#     arr.append(tn)
#     arr.append(fp)
#     arr.append(fn)
#     arr.append(tp)
#     score = (tn + tp) / (tn + fp + fn + tp)

#     f.write(chosen[i] + ": " + str(arr) + "   " + str(score) + "\n")

#     print("Feature : {} ".format(chosen[i]))
#     print(score)


def add_to_df(path, ch, fd):
    global da, de, rec, co, ya
    if os.path.exists(path):
        df = pd.read_csv(path, delimiter=',', header=None)
        df[8] = df[8].replace({';': ''}, regex=True)

        for l in range(9):
            if str(l) not in "14678":
                if ch == 'a':
                    da[arr[l]].append(
                        pd.Series([float(z) for z in df.iloc[:, l].values], dtype='object'))
                else:
                    de[arr[l]].append(
                        pd.Series([float(z) for z in df.iloc[:, l].values], dtype='object'))
        rec[co] = ["{}{}{}{}".format(ch, i, j, k), fd]
        co += 1
        if ch == 'a':
            ya.append(fd)
        else:
            ye.append(fd)

#Subject, Activity , Trial_no
#young & old
# 21 , 20 , 4 ( 3 trials )


for i in range(1, 21):
    for j in range(1, 20):
        for k in range(1, 4):

            # subject
            a = str(i)
            if len(a) == 1:
                subject = '0' + a
            else:
                subject = a
            # Activity
            b = str(j)
            if len(b) == 1:
                activity = '0' + b
            else:
                activity = b
            # Trial_no
            c = str(k)
            if len(c) == 1:
                trial = '0' + c
            else:
                trial = c

            path = "{}SA{}/D{}_SA{}_R{}.txt".format(
                s, subject, activity, subject, trial)
            # format i, j , k | F/D
            add_to_df(path, 'a', 'D')

            path = "{}SA{}/F{}_SA{}_R{}.txt".format(
                s, subject, activity, subject, trial)
            add_to_df(path, 'a', 'F')


da = pd.DataFrame(da)

# print(da.head)

for i in range(1, 21):
    for j in range(1, 20):
        for k in range(1, 4):

            # subject
            a = str(i)
            if len(a) == 1:
                subject = '0' + a
            else:
                subject = a
            # Activity
            b = str(j)
            if len(b) == 1:
                activity = '0' + b
            else:
                activity = b
            # Trial_no
            c = str(k)
            if len(c) == 1:
                trial = '0' + c
            else:
                trial = c

            path = "{}SE{}/D{}_SE{}_R{}.txt".format(
                s, subject, activity, subject, trial)
            # format i, j , k | F/D
            add_to_df(path, 'e', 'D')

            path = "{}SE{}/F{}_SE{}_R{}.txt".format(
                s, subject, activity, subject, trial)
            add_to_df(path, 'e', 'F')


print(de)

de = pd.DataFrame(de)


# ye = pd.Series(ye)
frame = [da, de]
X = pd.concat(frame)


# pad DF to produce consistent size each data sample
pad = PaddingTransformer().fit(X)
X = pad.transform(X)
print(X.shape)

# print(X.to_numpy())

Y = pd.Series(ya + ye)


# perform K fold cross validation on time series data

# arr = {"optKfold": [0, 0]}

# 5-Fold CV
tscv = TimeSeriesSplit(n_splits=5)

# ma = - 1
# opt_arr = []


# K fold cross validation
for train, test in tscv.split(X):
    print("%s %s" % (train, test))
    f.write("\n")
    X_train, X_test = X.loc[train], X.loc[test]
    y_train, y_test = Y.loc[train], Y.loc[test]

    TimeSeriesF(X_train, y_train, X_test, y_test)


f.close()


# f = open("TSFScores.txt", 'w')
# # K fold cross validation
# for train, test in tscv.split(X):
#     print("%s %s" % (train, test))
#     X_train, X_test = X.loc[train], X.loc[test]
#     y_train, y_test = Y.loc[train], Y.loc[test]

#     score, score_arr, clf = TimeSeriesF(X_train, y_train, X_test, y_test)
#     f.write(str(score) + " " + str(score_arr) + "\n\n")
#     if score > ma:
#         ma = score
#         arr["optKfold"][0] = train
#         arr["optKfold"][1] = test
#         opt_arr = score_arr
#         fe = 'TSFmodelKfold.sav'
#         pickle.dump(clf, open(fe, 'wb'))


# print(opt_arr, ma)
# f.close()

# X_train, X_test, y_train, y_test = train_test_split(X,  Y, random_state=42)

# print(type(X_train))

# K_nn(X_train, y_train, X_test, y_test)


# # RISE
# rise(X_train, y_train, X_test, y_test)


# # K_nn(X_train, X_test, y_train, y_test)


# print("hello")
