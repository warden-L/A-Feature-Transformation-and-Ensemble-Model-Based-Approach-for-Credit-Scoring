import pandas as pd
import numpy as np
import xlearn as xl
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.layers import Dense, embeddings
from sklearn import metrics
import os
import tensorflow.compat.v1 as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, log_loss, precision_score, recall_score

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from xgboost import XGBClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
or_data = pd.read_csv("../cs-training.csv")
label = or_data["label"]
or_data = or_data.drop(["ID", "label"], axis=1)
ae_data = pd.read_csv("../cs_ALEX_test_residue.csv").drop(["label"], axis=1)
xgb_data = pd.read_csv("cs_XGB_FT.csv")
or_data.columns = ["Original_" + str(f) for f in range(len(or_data.columns))]
ae_data.columns = ["AE_" + str(f) for f in range(len(ae_data.columns))]
xgb_data.columns = ["BT_" + str(f) for f in range(len(xgb_data.columns))]


class OneHot:
    def __init__(self, df, column_name_list):
        self.df = df
        self.column_name_list = column_name_list

    def multi_column_encoder(self):
        Enc_ohe, Enc_label = OneHotEncoder(), LabelEncoder()
        for column_name in self.column_name_list:
            self.df["Dummies"] = Enc_label.fit_transform(self.df[column_name])
            self.df_dummies = pd.DataFrame(Enc_ohe.fit_transform(self.df[["Dummies"]]).todense(),
                                           columns=Enc_label.classes_)
            self.df_dummies.rename(columns=lambda x: str(column_name) + "_" + str(x), inplace=True)
            # print('df[[Dummies]]:', self.df[["Dummies"]].shape)
            # print('df[Dummies]:', self.df["Dummies"].shape)
            # print('df_dummies:', self.df_dummies.shape)
            # print('df:', self.df.shape)
            name = self.df.columns.values.tolist() + self.df_dummies.columns.values.tolist()
            # print(self.df_dummies)
            self.df.reset_index(drop=True, inplace=True)
            self.df_dummies.reset_index(drop=True, inplace=True)
            self.df = pd.concat([self.df, self.df_dummies], axis=1, ignore_index=True)
            self.df.columns = name
        self.df.drop(["Dummies"], axis=1, inplace=True)
        self.df.drop(self.column_name_list, axis=1, inplace=True)
        return self.df


# xgb_data = OneHot(xgb_data, xgb_data.columns.values).multi_column_encoder()
# xgb_data.to_csv("cs_xgb_onehot.csv", index=False)
xgb_data = pd.read_csv("cs_XGB_FT.csv")
print(xgb_data.shape)
for feat in xgb_data.columns.values:
    lbe = LabelEncoder()
    xgb_data[feat] = lbe.fit_transform(xgb_data[feat])
col = xgb_data.columns
xgb_data = np.array(xgb_data)
max = np.max(xgb_data)
model = Sequential()
# model.add(xgb_data.shape[1], 4)
model.add(embeddings.Embedding(max, 4, input_length=xgb_data.shape[1]))
input_array = xgb_data
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)

xgb_data = pd.DataFrame(output_array.reshape(xgb_data.shape[0], -1),
                        columns=["BT" + str(f) for f in range(output_array.shape[1])])
data = pd.concat([or_data, ae_data, xgb_data], axis=1)
data = data.fillna(-1, )
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=21, shuffle=True)

nohl = [400, 400, 400, 400]  # number of neurons in each hidden layer

ANN = Sequential()


def AUC(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# Hidden Layer
for i in range(len(nohl)):
    if i == 0:
        ANN.add(
            Dense(units=nohl[i], input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
    else:
        ANN.add(Dense(units=nohl[i], kernel_initializer=glorot_uniform(seed=0), activation='relu'))

# Output Layer
ANN.add(Dense(units=1, kernel_initializer=glorot_uniform(seed=0), activation='sigmoid'))

ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC, 'binary_crossentropy'])
earlyStopping = EarlyStopping(monitor='val_AUC', patience=50, verbose=0, min_delta=1e-4, mode='auto')
lr_reduced = ReduceLROnPlateau(monitor='val_AUC', factor=0.5, patience=50, verbose=0, epsilon=1e-4, mode='auto')
ANN.fit(X_train, y_train, epochs=1000, validation_split=0.1, batch_size=4096,
        callbacks=[earlyStopping, lr_reduced], shuffle=True)
pred = ANN.predict_proba(X_test)

plt.figure(0).clf()
pp1 = []
for i in pred:
    pp1.append(i[0])
p_int = [i >= 0.5 for i in pp1]
a1 = roc_auc_score(y_test, pp1)
logloss = log_loss(y_test, pp1)
annacc = accuracy_score(y_test, p_int)

xgb_parma = {}
model = XGBClassifier(**xgb_parma, n_estimators=1000)

model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], eval_metric="auc")
pre2 = model.predict_proba(X_test)[:, 1]
pre_int = (pre2 > 0.5) * 1
s = cross_val_score(model, data, label, cv=10, scoring="roc_auc")
l = cross_val_score(model, data, label, cv=10, scoring="neg_log_loss")

fm_parma = {}
fm = xl.FMModel()
fm.fit(X_train, y_train)
pre = fm.predict(X_test)

fm_p = pd.DataFrame(pre)
dnn_p = pd.DataFrame(pp1)
data2 = pd.concat([fm_p, dnn_p], axis=1)
label2 = y_test
lr = LogisticRegression()
s = cross_val_score(lr, data2, label2, cv=10, scoring="roc_auc")
l = cross_val_score(lr, data2, label2, cv=10, scoring="neg_log_loss")
print("cv auc:", np.mean(s))
print("cv logloss:", np.mean(l))
