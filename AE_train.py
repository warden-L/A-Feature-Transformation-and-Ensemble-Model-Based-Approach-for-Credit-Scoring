import os
import random
import tensorflow.compat.v1 as tf
from keras.callbacks import *
from keras.layers import *

from keras.optimizers import adam
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import *

import matplotlib.pyplot as plt

seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
tf.keras.backend.set_session(sess)


def ResidueMaker(save, name, model, data_X, data_Y, sheet_head):
    data_predicted = model.predict(data_X)
    data_residue = data_Y - data_predicted
    data_residue = Tensor3Dto2D(data_residue)
    data_residue_df = pd.DataFrame(columns=sheet_head, data=data_residue)
    if save:
        data_residue_df.to_csv(name, index=0)
    return data_residue_df


def AUC(Attack_Label, Test_Residue):
    scaler = MinMaxScaler()
    Normalized_Test_Residue = scaler.fit_transform(np.abs(Test_Residue))
    auc = np.zeros(Test_Residue.shape[1])
    for i in range(Test_Residue.shape[1]):
        auc[i] = roc_auc_score(Attack_Label, Normalized_Test_Residue[:, i])
    return auc


def Tensor3Dto2D(Tensor3D):
    Sample_Number = Tensor3D.shape[0]
    Sample_Row = Tensor3D.shape[1]
    Sample_Column = Tensor3D.shape[2]
    Tensor2D = Tensor3D.reshape(Sample_Number * Sample_Row, Sample_Column)
    return Tensor2D


def Tensor2Dto3D(Tensor2D, Sample_Row):
    Sample_Range = Tensor2D.shape[0]
    Sample_Column = Tensor2D.shape[1]
    Tensor3D = Tensor2D[0:Sample_Range // Sample_Row * Sample_Row, :].reshape(-1, Sample_Row, Sample_Column)
    return Tensor3D


def ResidueMaker(save, name, model, data, sheet_head):
    data_predicted = model.predict(data)
    data_residue = data - data_predicted
    data_residue = Tensor3Dto2D(data_residue)
    data_residue_df = pd.DataFrame(columns=sheet_head, data=data_residue)
    if save:
        data_residue_df.to_csv(name, index=0)
    return data_residue_df


class AEED:
    def __init__(self, nH, cf, activation):
        self.nI = None  # number of inputs；the number of sensors
        self.nH = nH  # number of hidden layers in encoder (decoder)
        self.cf = cf  # compression factor
        self.activation = activation  # Auto-Encoder activation function
        self.model = None

    def create_model(self, Training_x):
        self.nI = Training_x.shape[2]
        temp = np.linspace(self.nI, self.nI / self.cf, self.nH + 1).astype(
            int)  # array([117, 101,  86,  71,  56,  41])——ALEX.J.LEE 20201003
        nH_enc = temp[1:]  # array([101,  86,  71,  56,  41])——ALEX.J.LEE 20201003
        nH_dec = temp[:-1][::-1]  # array([ 56,  71,  86, 101, 117])——ALEX.J.LEE 20201003
        encoder_inputs = Input(
            shape=(Training_x.shape[1], Training_x.shape[2]))  # TensorShape([None, 1, 117])——ALEX.J.LEE 20201003
        for i, layer_size in enumerate(nH_enc):
            if i == 0:
                encoder = Dense(layer_size, activation=self.activation)(encoder_inputs)
            else:
                encoder = Dense(layer_size, activation=self.activation)(encoder)
        for i, layer_size in enumerate(nH_dec):
            if i == 0:
                decoder = Dense(layer_size, activation=self.activation)(encoder)
            else:
                decoder = Dense(layer_size, activation=self.activation)(decoder)
        model = Model(inputs=encoder_inputs, outputs=decoder)
        self.model = model
        model.summary()
        model.compile(loss='mse', optimizer=adam(lr=0.001))
        return model


def Train(model, Training_x, Training_y, Validation_x, Validation_y, model_name):
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, min_delta=1e-4, mode='auto')
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, epsilon=1e-4, mode='min')
    train_start = time.time()
    history = model.fit(Training_x, Training_y,
                        epochs=2,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[earlyStopping, lr_reduced],
                        verbose=2,
                        validation_data=(Validation_x, Validation_y)
                        )
    train_end = time.time()
    print("Runing Time for traning:", train_end - train_start)
    model.save("../model/" + model_name + '.h5')
    return history


def draw(history, model_name):
    fig = plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title(model_name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig("../model/" + model_name + '_loss.png')


if __name__ == '__main__':
    batch_size = 32
    sample_row = 1
    nH = 5  # number of hidden layers in encoder (decoder)
    cf = 2  # compression factor
    activation = 'tanh'  # autoencoder activation function

    Path_Training_Data = "Train_preprocessed.csv"
    Path_Testing_Data = "Test_preprocessed.csv"

    Training_data = pd.read_csv(Path_Training_Data)
    Testing_data = pd.read_csv(Path_Testing_Data)
    Training_Norm, Validation_Norm, _, _ = train_test_split(Training_data, np.zeros(Training_data.shape[0]),
                                                            test_size=0.33, shuffle=False)
    name = Training_data.columns
    Training_Norm = Training_Norm.values
    Validation_Norm = Validation_Norm.values
    Testing_data = Testing_data.values
    Testing_Norm = Testing_data[:, :Testing_data.shape[1] - 1]
    # Reconstructed
    Training_Data_3DTensor = Tensor2Dto3D(Training_Norm, sample_row)
    Validation_Data_3DTensor = Tensor2Dto3D(Validation_Norm, sample_row)
    Testing_Data_3DTensor = Tensor2Dto3D(Testing_Norm, sample_row)
    Training_x = Training_Data_3DTensor
    Training_y = Training_Data_3DTensor
    Validation_x = Validation_Data_3DTensor
    Validation_y = Validation_Data_3DTensor
    Testing_x = Testing_Data_3DTensor
    Testing_y = Testing_Data_3DTensor
    # AEED
    model_name = 'AEED-4-3'
    nI = Training_x.shape[2]
    autoencoder = AEED(nH, cf, activation)
    model = autoencoder.create_model(Training_x)
    history = Train(model, Training_x, Training_y, Validation_x, Validation_y, model_name)

    AEED_model = load_model("../model/" + model_name + '.h5')
    # AEED = load_model(' bb.h5', custom_objects={'MemoryUnit': MemoryM.MemoryUnit, 'loss': MemoryM.Normentropy},compile=False)

    Valid_Residue = ResidueMaker(False, 'AE_validation_residue.csv', AEED_model, Validation_x, Validation_y,
                                 name)
    Test_Residue = ResidueMaker(False, 'AE_test_residue.csv', AEED_model, Testing_x, Testing_y, name)
