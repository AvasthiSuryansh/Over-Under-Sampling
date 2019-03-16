from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 5)

def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f1: {}".format(f1_score(true_value, pred)))


# classifier we are using
classifier = RandomForestClassifier

data = pd.read_csv(r'/Users/suryanshavasthi/Downloads/creditcard.csv')
#data = pd.read_csv(r'/Users/suryanshavasthi/Downloads/winequality/winequality-white.csv')
#data = fetch_datasets()['/Users/suryanshavasthi/Downloads/data.csv']

print(data.head())

X_train, X_test, y_train, y_test = train_test_split(data['Amount'].values.reshape(-1,1), data['Class'].values.reshape(-1,1), random_state=2)

# building a normal model
pipeline = make_pipeline(classifier(random_state=42))
model = pipeline.fit(X_train, y_train)
prediction = model.predict(X_test)

# building a model using over sampling technique (SMOTE) in imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), classifier(random_state=42))
smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)

# building a model using under sampling technique (NearMiss) in imblearn
nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state=42), classifier(random_state=42))
nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
nearmiss_prediction = nearmiss_model.predict(X_test)

# printing the information of all the 3 models
print()
print("normal data distribution: {}".format(Counter(data['Class'])))
X_smote, y_smote = SMOTE().fit_sample(data['Amount'].values.reshape(-1,1), data['Class'].values.reshape(-1,1))
print("SMOTE data distribution: {}".format(Counter(y_smote)))
X_nearmiss, y_nearmiss = NearMiss().fit_sample(data['Amount'].values.reshape(-1,1), data['Class'].values.reshape(-1,1))
print("NearMiss data distribution: {}".format(Counter(y_nearmiss)))


# classification report
print(classification_report(y_test, prediction))
print(classification_report_imbalanced(y_test, smote_prediction))

print()
print('normal Pipeline Score {}'.format(pipeline.score(X_test, y_test)))
print('SMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))
print('NearMiss Pipeline Score {}'.format(nearmiss_pipeline.score(X_test, y_test)))


print()
print_results("normal classification", y_test, prediction)
print()
print_results("SMOTE classification", y_test, smote_prediction)
print()
print_results("NearMiss classification", y_test, nearmiss_prediction)




