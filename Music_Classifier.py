#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:44:15 2021
EEE498/591 Final Project
Naive Boyes
@author: Adan, Jonah, Zach, Dale
"""
# import librosa # used to preprocess data (MFCCs, spectrograms), not used in this file because data.csv already has features extracted
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd  # used to read data from GTZAN dataset
from sklearn import linear_model
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  # used for knn classifier
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# used to get accuracy metrics for models
from sklearn.metrics import accuracy_score, confusion_matrix
# import for plotting with matplotlib and seaborn
from matplotlib import pyplot as plt
import joblib
import seaborn as sn
# import for time
import time

# used to read data from GTZAN dataset for feature extraction
data = pd.read_csv('data.csv')
data = data.drop(['filename'], axis=1)  # drops the file name from the dataset

# used to drop the mfccs from 13 to 20 for training the model
# dropped
data = data.drop(['zero_crossing_rate'], axis=1)
data = data.drop(['mfcc13'], axis=1)
data = data.drop(['mfcc14'], axis=1)
data = data.drop(['mfcc15'], axis=1)
data = data.drop(['mfcc16'], axis=1)
data = data.drop(['mfcc17'], axis=1)
data = data.drop(['mfcc18'], axis=1)
data = data.drop(['mfcc19'], axis=1)
data = data.drop(['mfcc20'], axis=1)

genre_list = ["Blues", "classical", "country", "disco",
              "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# used to plot the confusion matrix
def plot_matrix(cm, title, genre):
    df_cm = pd.DataFrame(cm, index=["Blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                         columns=["Blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
    # plt.xlim(-0.5, len(np.unique(y))-0.5)
    # plt.ylim(len(np.unique(y))-0.5, -0.5)
    plt.figure(figsize=(13, 10)),	plt.title(title)
    sn.set(font_scale=3)
    # have to add in the limits in order to correct for matplotlib and seaborn version
    corrected = sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
    bottom, top = corrected.get_ylim()
    corrected.set_ylim(bottom+0.5, top-0.5)
    plt.show()


# used for polynomial classifier
poly_params = {
    "cls__C": [0.5, 1, 2, 5],
    "cls__kernel": ['poly'],
}


pipe_svm = Pipeline([
    ('scale', StandardScaler()),
    ('var_tresh', VarianceThreshold(threshold=(.8 * (1 - .8)))),
    ('feature_selection', SelectFromModel(lgbm.LGBMClassifier())),
    ('cls', svm.SVC())
])


###  TRAINING DATASET W/ ALL FEATURES  ###

# Scaling the dataset
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

# Perform Principal Component Analysis
pca = PCA()
X = pca.fit_transform(X)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# used to split the dataset into train and test sets, using regular 30% for test set size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# ### CLASSIFICATIONS ###

# #Training Model using KNN
tbeg = time.time()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# used to find the best k value for knn
grid_params = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17],
    "weights": ["uniform"],
    "metric": ["euclidean", "manhattan"]
}
grid_knn = GridSearchCV(KNeighborsClassifier(),
                        grid_params, verbose=1, cv=5, n_jobs=-1)
grid_knn.fit(X_train, y_train)
tend = time.time()
knn_pred = grid_knn.predict(X_test)

# used to calculate and print the accuracy scores
print()
print("KNN Accuracy Metrics:")
print("Train set accuracy: {:.2f}".format(grid_knn.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(accuracy_score(y_test, knn_pred)))
print('Best n_neighbors:',
      grid_knn.best_estimator_.get_params()['n_neighbors'])
print('total training time %.4f  ' % (tend-tbeg))
print()

# used to plot confusion matrix for knn
knn_cm = confusion_matrix(y_test, knn_pred)
plot_matrix(knn_cm, "KNN", genre_list)
grid_cm = pd.DataFrame(confusion_matrix(y_test, knn_pred), index=["Blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                       columns=["Blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
plt.figure(figsize=(13, 10)),	plt.title("KNN")
sn.set(font_scale=3)
# have to add in the limits in order to correct for matplotlib and seaborn version
corrected = sn.heatmap(grid_cm, annot=True, cmap="YlGnBu")
bottom, top = corrected.get_ylim()
corrected.set_ylim(bottom+0.5, top-0.5)
# sn.heatmap(grid_cm, annot=True, cmap="PiYG")


# Training model using SVM - support vector machine
# Function to train the model using SVM
def svm_model(params, X_train, y_train, X_test, y_test, title):
    tbeg = time.time()
    svm = GridSearchCV(pipe_svm, params, scoring='accuracy', cv=5)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    tend = time.time()
    train_accuracy = svm.score(X_train, y_train)
    test_accuracy = svm.score(X_test, y_test)
    print(" SVM Accuracy Metrics:")
    print("Train set accuracy: {:.2f}".format(train_accuracy))
    print("Test set accuracy: {:.2f}".format(test_accuracy))
    print('total training time %.4f  ' % (tend-tbeg))
    print()
    svm_cm = confusion_matrix(y_test, svm_pred)
    plot_matrix(svm_cm, title, genre_list)


svm_model(poly_params, X_train, y_train, X_test, y_test, "Polynomial SVM")


# Training model using Logistic Regression
# used to train logistic regression model
def log_reg_func(X_train, y_train, X_test, y_test, genre):
    tbeg = time.time()
    logistic_classifier = linear_model.LogisticRegression(max_iter=1000)
    logistic_classifier.fit(X_train, y_train)
    logistic_predictions = logistic_classifier.predict(X_test)
    tend = time.time()
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    logistic_cm = confusion_matrix(y_test, logistic_predictions)
    print("Logistic Regression Accuracy Metrics:")
    print("Train set accuracy: {:.2f}".format(
        logistic_classifier.score(X_train, y_train)))
    print("Test set accuracy: {:.2f}".format(logistic_accuracy))
    print('total training time %.4f  ' % (tend-tbeg))
    print()
    joblib.dump(logistic_classifier, 'model.pkl')
    plot_matrix(logistic_cm, "Logistic Regression", genre)


log_reg_func(X_train, y_train, X_test, y_test, genre_list)


# Training model using Random Forest
from sklearn.ensemble import RandomForestClassifier
def random_forest(X_train, y_train, X_test, y_test, genre):
    # PARAMETER TUNING
    tbeg = time.time()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
    rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
    rf_random.fit(X_train, y_train)

# rf_random.best_params
    forest_predictions = rf_random.predict(X_test)

    ran_forest = []
    tbeg = time.time()
    for i in range(2,40):
        forest=RandomForestClassifier(random_state=42,n_estimators=i)
        forest.fit(X_train,y_train)
        ran_forest.append(forest.score(X_test, y_test))
        max_accuracy = max(ran_forest)
        best_n_est=2+ran_forest.index(max(ran_forest))

    print("Random Forest Accuracy Metrics:")
    print("Max Accuracy is {:.3f} on test dataset with {} estimators".format(max_accuracy,best_n_est))
    plt.plot(np.arange(2,20),ran_forest) # this plots the accuracy vs. n_estimators
    plt.xlabel("n Estimators")
    plt.ylabel("Accuracy")
    forest=RandomForestClassifier(random_state=42,n_estimators=best_n_est, max_features =20, max_depth=100, min_samples_leaf=2)
    forest=RandomForestClassifier(random_state=42,n_estimators=best_n_est,max_depth=200)
    forest.fit(X_train,y_train)
    forest_predictions = forest.predict(X_test)
    tend = time.time()
    # forest_accuracy = accuracy_score(y_test, forest_predictions)
    print("Training set accuracy: {:.3f}".format(
        rf_random.score(X_train, y_train)))
    print("Test set accuracy: {:.2f}".format(rf_random.score(X_test, y_test)))
    # print("Test score: {:.3f}".format(forest.score(X_test,y_test)))
    print('total training time %.4f  ' % (tend-tbeg))
    print()
    forest_cm = confusion_matrix(y_test, forest_predictions)
    plot_matrix(forest_cm, "Random Forest", genre)
random_forest(X_train, y_train, X_test, y_test, genre_list)


# Training model using Neural Network


def multilayer_perc(X_train, y_train, X_test, y_test, genre):
    tbeg = time.time()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10),
                        early_stopping=(True), learning_rate='adaptive', random_state=2, max_iter=2000)
    clf.fit(X_train, y_train)
    MLP_predict = clf.predict(X_test)
    tend = time.time()
    print("Multilayer Perceptron Metrics: ")
    print("Training set accuracy: {:.3f}".format(clf.score(X_train, y_train)))
    print("Test set accuracy: {:.2f}".format(
        accuracy_score(y_test, MLP_predict)))
    # print("Test set accuracy 2: {:.2f}".format(clf.score(y_test,MLP_predict)))
    print('total training time %.4f  ' % (tend-tbeg))
    print()
    MLP_cm = confusion_matrix(y_test, MLP_predict)
    plot_matrix(MLP_cm, "Multilayer Perceptron", genre)


multilayer_perc(X_train, y_train, X_test, y_test, genre_list)


def naive_bayes(X_train, y_train, X_test, y_test, genre):
    from sklearn.naive_bayes import GaussianNB
    tbeg = time.time()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    NB_predict = gnb.predict(X_test)
    tend = time.time()
    NB_accuracy = accuracy_score(y_test, NB_predict)
    print('Naive Bayes Accuracy Metrics')
    print("Training set accuracy for Gaussian: {:.3f}".format(
        gnb.score(X_train, y_train)))
    print("Test set accuracy: {:.2f}".format(NB_accuracy))
    print('total training time %.4f  ' % (tend-tbeg))
    print()
    NB_cm = confusion_matrix(y_test, NB_predict)
    plot_matrix(NB_cm, "Naives Bayes", genre)
    from sklearn.model_selection import cross_val_score
    print('cross val score:', cross_val_score(gnb, X_test, y_test, cv=5))


naive_bayes(X_train, y_train, X_test, y_test, genre_list)


# Training Model using DecisionTree


def decisionTree(X_train, y_train, X_test, y_test, genre):
    tree_model = tree.DecisionTreeClassifier()
    tbeg = time.time()
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)
    tend = time.time()
    score_tree = accuracy_score(y_test, tree_pred)
    DT_cm = confusion_matrix(y_test, tree_pred)
    plot_matrix(DT_cm, "Decision Tree", genre)
    print("Decision Tree Accuracy Metrics:")
    print("Train set accuracy: {:.2f}".format(
        tree_model.score(X_train, y_train)))
    print("Test set accuracy: {:.2f}".format(score_tree))
    print('total training time %.4f  ' % (tend-tbeg), "\n")


decisionTree(X_train, y_train, X_test, y_test, genre_list)


# need to solve the overfifitting issue in a with a neural network

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """
    fig, axs = plt.subplots(2)
    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    plt.show()


def NN_solve(X, X_train, y_train, X_test, y_test, genre):
    #from keras.models import Sequential
    import tensorflow.keras as keras

    # need to add in a new axis to each to make the data 3D
    X_trainn = X_train[..., np.newaxis]
    X_testt = X_test[..., np.newaxis]
    y_trainn = y_train[..., np.newaxis]
    y_testt = y_test[..., np.newaxis]
    X = X[..., np.newaxis]
    # time start
    tbeg = time.time()
    # create the sequential model
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
        # 1st dense layer
        keras.layers.Dense(512, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),
        # 2nd dense layer
        keras.layers.Dense(256, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd dense layer
        keras.layers.Dense(64, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # train model
    history = model.fit(X_trainn, y_trainn, validation_data=(
        X_testt, y_testt), batch_size=32, epochs=300)
    # plot accuracy and error as a function of the epochs
    plot_history(history)

    NN_pred_tr = model.predict(X_train)
    NN_pred_tr = np.argmax(NN_pred_tr, axis=1)
    NN_pred = model.predict(X_test)
    tend = time.time()
    NN_pred = np.argmax(NN_pred, axis=1)
    # accuracy_train = accuracy_score(y_train, NN_pred_tr)
    # accuracy_train = model.evaluate(X_test, y_test)
    # accuracy_test = model.evaluate(X_train,y_train)
    accuracy_test = accuracy_score(y_test, NN_pred)
    # print("Train set accuracy: {:.2f}".format(accuracy_train))
    print("Test set accuracy: {:.2f}".format(accuracy_test))
    print('total training time %.4f  ' % (tend-tbeg), "\n")
    # print(model.evaulate(X_train,y_train))
    # print(model.evaulate(X_test,y_test))
    NN_cm = confusion_matrix(y_test, NN_pred)
    plot_matrix(NN_cm, "Neural Network w/ Dropout + Regularization", genre)


NN_solve(X, X_train, y_train, X_test, y_test, genre_list)
