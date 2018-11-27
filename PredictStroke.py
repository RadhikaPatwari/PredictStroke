import pandas as pd  # to import csv and for data manipulation
import matplotlib.pyplot as plt  # to plot graph
import seaborn as sns  # for intractve graphs
import numpy as np  # for linear algebra
import datetime  # to dela with date and time
from sklearn.preprocessing import StandardScaler, LabelEncoder  # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier
from sklearn.tree import DecisionTreeClassifier  # for Decision Tree classifier
from sklearn.svm import SVC  # for SVM classification
import sklearn.linear_model as sk #import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import \
    GridSearchCV  # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import \
    RandomizedSearchCV  # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix, recall_score, precision_recall_curve, auc, roc_curve, roc_auc_score, \
    classification_report
import warnings
import pickle
import sys
import os




    data = pd.read_csv("C:/Users/PWR5KOR/Downloads/healthcare-dataset-stroke-data/train_2v.csv", header=0)
    data.info()
    print(data.describe())
    print('dataset shape: {}'.format(data.shape))
    # print the first 2 rows of data
    print(data.head(2))





    print((data[["bmi"]] == 0).sum())
    print((data[["avg_glucose_level"]] == 0).sum())
    data['bmi'].replace(0, np.nan, inplace=True)
    data['avg_glucose_level'].replace(0, np.nan, inplace=True)

    # count the number of NaN values in each column
    print(data.isnull().sum())




    # fill missing values with mean column values
    data.fillna(data.mean(), inplace=True)
    data.dropna(inplace=True)

    print(data.isnull().sum())
    print('dataset shape: {}'.format(data.shape))

    #data.to_csv('C:/Users/PWR5KOR/Desktop/Predict_Stroke/cleandata_string.csv', index=False)

    le = LabelEncoder()

    for column_name in data.columns:
        if data[column_name].dtype == object:
            data[column_name] = le.fit_transform(data[column_name])
        else:
            pass

    #data.to_csv('C:/Users/PWR5KOR/Desktop/Predict_Stroke/cleandata.csv', index=False)


    #---------------------------------------------------------------------------------------------------------------

    # print(data.stroke.value_counts())
    #
    # count_class_0, count_class_1 = data.stroke.value_counts()
    # # Divide by class
    # df_class_0 = data[data['stroke'] == 0]
    # df_class_1 = data[data['stroke'] == 1]
    #
    # df_class_1_over = df_class_1.sample(int(count_class_0 / 4), replace=True)
    # data_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    #
    # print('Random over-sampling:')
    # print(data_over.stroke.value_counts())


    #----------------------------------------------------------------------------------------------------------------

    # split dataset in features and target variable
    feature_cols = ['age', 'hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level', 'bmi',
                    'smoking_status']
    X = data[feature_cols]  # Features
    y = data.stroke  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #logreg_smoke = sk.LogisticRegression()
    logreg_smoke = RandomForestClassifier()
    logreg_smoke.fit(X_train, y_train)

    y_pred = logreg_smoke.predict(X_test)
    print(y_pred)

    # Accuracy
    print('Accuracy of Model is:- ', logreg_smoke.score(X_test, y_test))

    # Confusion matrix
    # confusion_matrix = confusion_matrix(y_test, y_pred)
    # print(confusion_matrix)

    # ROC Curve
    y_pred_proba = logreg_smoke.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()

    #-----------------------------------------------------------------------------------------------------------------
    # Validating
    # data1 = pd.read_csv("C:/Users/PWR5KOR/Desktop/validate.csv", header=0)
    #
    # y_pred_validate = logreg_smoke.predict(data1[feature_cols])
    # print(y_pred_validate)
    #
    # data['stroke1'] = y_pred_validate
    # print(data)
    #
    # confusion_matrix = confusion_matrix(data1.stroke, y_pred_validate)
    # print(confusion_matrix)
    #
    # data.to_csv('C:/Users/PWR5KOR/Desktop/Predict_Stroke/output.csv', index=False)



    #------------------------------------------------------------------------------------------------------------------
    #dumping model
    # serialize the model on disk
    print("Export the model to.pkl")
    f1 = open('./stroke.pkl', 'wb')
    pickle.dump(logreg_smoke, f1)
    f1.close()







