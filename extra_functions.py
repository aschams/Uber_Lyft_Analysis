import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def baglr(lr: LogisticRegression,
          X0: pd.DataFrame,
          X1: pd.DataFrame,
          X_test: pd.DataFrame,
          y_test: np.array ,
          features: list,
          n: int,
          nloop: int):
    """
    This function take in the logistic regression, list of features, number of samples, and number of loops as arguments,
    resample the two subset of dataframes, and return the score and predictions as lists. It also return the coefficients
    of the model with the highest score in the loop.
    Inputs:
        lr: LogisticRegression Object
        X0: DataFrame of data containing negative class
        X1: DataFrame of data containing positive class
        X_test: DataFrame of training data
        y_test: DataFrame of test data
        features: names of features to extract from X to build the models
        n: size of resample
        nloop: Number of times to repeat bagging process
    Outputs:
        lrcoefs: coefficients of the best logistic regression model found
        lrscore: scores of the logistic regression models built
        lrpred: predictions of all of the logistic regression models built
    """
    lrcoeff = []
    lrscore = []
    lrpred = []
    for i in range(nloop):
        mySample1 = X1.sample(n, replace=True, random_state=i)
        mySample = mySample1.append(X0.sample(n, replace=True))
        mySample.reset_index(inplace=True, drop=True)
        y_sample = mySample['y']
        x_sample = mySample[features]
        lr.fit(x_sample, y_sample)
        print(lr.score(x_sample, y_sample), '\n', lr.coef_)
        lrscore.append(copy.deepcopy(lr.score(x_sample, y_sample)))
        lrcoeff.append(copy.deepcopy(lr.coef_))
        lrpred.append(copy.deepcopy(lr.predict(X_test)))
        probs = lr.predict_proba(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    besti = lrscore.index(max(lrscore))
    lrcoefs = {}
    for i in range(0, len(features)):
        lrcoefs.update({features[i]: lrcoeff[besti][0][i]})
    return lrcoefs, lrscore, lrpred

def get_avg_by_cab(df: pd.DataFrame,
                   cab_type: str,
                   gb_feature: str,
                   feature: str) -> pd.DataFrame:
    """
    Returns a dataframe for the average value of a given feature based on a groupby feature.
    Inputs:
        df: Source DataFrame
        cab_type: Type of cab; 'Lyft' or 'Uber'
        gb_feature: feature to groupby df
        feature: feature to take the mean of
    Outputs:
        avg_price: df of the average value of feature, grouped by gb_feature
    """
    assert cab_type == 'Lyft' or cab_type == 'Uber','Invalid cab_type specified'
    avg_by_timestamp = df[df.cab_type == cab_type].groupby(by=gb_feature).mean()
    avg = avg_by_timestamp[feature]
    avg.index = pd.to_datetime(avg.index)
    avg_price = avg.groupby(pd.Grouper(freq='h')).ffill()
    return avg_price

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Taken from
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
