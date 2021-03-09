import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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
            self.df_dummies.rename(columns=lambda x: column_name + "_" + str(x), inplace=True)
            self.df = pd.concat([self.df, self.df_dummies], axis=1)
        self.df.drop(["Dummies"], axis=1, inplace=True)
        self.df.drop(self.column_name_list, axis=1, inplace=True)
        return self.df


class Preprocessing:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def sparse_onehot_dense_minmax(self, denses_name, sparses_name):
        dense_cols = [col for col in self.df_train.columns if col in denses_name]
        sparse_cols = [col for col in self.df_train.columns if col in sparses_name]

        Training_sparse = pd.DataFrame(index=self.df_train.index, columns=sparse_cols,
                                       data=self.df_train[
                                           sparse_cols])
        Testing_sparse = pd.DataFrame(index=self.df_test.index, columns=sparse_cols,
                                      data=self.df_test[sparse_cols])

        Training_sparse_onehot = OneHot(Training_sparse,
                                        sparses_name).multi_column_encoder()
        Testing_sparse_onehot = OneHot(Testing_sparse,
                                       sparses_name).multi_column_encoder()

        sparses_onehot_name_test = Testing_sparse_onehot.columns
        sparses_onehot_name_train = Training_sparse_onehot.columns
        Training_sparse_onehot_final = pd.DataFrame()

        for i in range(Testing_sparse_onehot.shape[1]):
            if (sparses_onehot_name_test[i] in sparses_onehot_name_train):
                Training_sparse_onehot_final[sparses_onehot_name_test[i]] = Training_sparse_onehot[
                    sparses_onehot_name_test[i]]
            else:
                Training_sparse_onehot_final[sparses_onehot_name_test[i]] = np.zeros(
                    Training_sparse_onehot.shape[0])
        Training_sparse_onehot = Training_sparse_onehot_final

        scaler = MinMaxScaler()
        Training_dense_Norm = pd.DataFrame(index=self.df_train.index, columns=dense_cols,
                                           data=scaler.fit_transform(
                                               self.df_train[dense_cols]))
        Testing_dense_Norm = pd.DataFrame(index=self.df_test.index, columns=dense_cols,
                                          data=scaler.transform(
                                              self.df_test[dense_cols]))

        Training_data = pd.concat([Training_sparse_onehot, Training_dense_Norm],
                                  axis=1)
        Testing_data = pd.concat([Testing_sparse_onehot, Testing_dense_Norm],
                                 axis=1)

        Testing_data['attack label'] = np.array(self.df_test.values[:, -1])
        return Training_data, Testing_data


if __name__ == '__main__':
    Path_Training_Data = ""
    Path_Testing_Data = ""
    sparses_name = []
    denses_name = []
    Training_Data_DataFrame = pd.read_csv(Path_Training_Data)
    Testing_Data_DataFrame = pd.read_csv(Path_Testing_Data)

    preprocessing_data = Preprocessing(Training_Data_DataFrame, Testing_Data_DataFrame)
    Training_data, Testing_data = preprocessing_data.sparse_onehot_dense_minmax(denses_name, sparses_name)
    print(Training_data.shape)
    print(Testing_data.shape)
    Training_data.to_csv('', index=0)
    Testing_data.to_csv('', index=0)
