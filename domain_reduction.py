import sys

sys.path.append('/home/jeremy.levy/Jeremy/copd_osa')

import pandas as pd
import os
from sklearn.decomposition import PCA
from icecream import ic
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard

from SHHS_Causal.preprocess_features import preprocess_data
from constants import x_columns, y_column, event_columns, t_baseline


def pca_reduction(spo2_features):
    pca = PCA(n_components=1)
    pca.fit(spo2_features)
    single_treatment = pca.transform(spo2_features)

    return single_treatment


def autoencoder_reduction(spo2_features):
    spo2_features = StandardScaler().fit_transform(spo2_features)

    input_dim = spo2_features.shape[1]
    encoding_dim = 1

    input_layer = Input(shape=(input_dim,))

    encoder = Dense(int(input_dim / 2), activation="tanh",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dense(int(input_dim / 2), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dense(int(input_dim / 4), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dense(int(input_dim / 4), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder_layer = Dense(encoding_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(encoder)

    decoder = Dense(int(input_dim / 4), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dense(int(input_dim / 4), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dense(int(input_dim / 2), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dense(int(input_dim / 2), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(decoder)

    output_layer = Dense(input_dim)(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=encoder_layer)
    autoencoder.compile(optimizer='adam', loss="mse", metrics=["mse", "mae"])

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.000001, verbose=1),
        EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True),
        TensorBoard(log_dir='logs', histogram_freq=1)
    ]

    autoencoder.fit(spo2_features, spo2_features, epochs=100, batch_size=8, shuffle=True, verbose=1,
                    callbacks=callbacks)

    autoencoder_treatment = encoder(spo2_features)
    return np.squeeze(autoencoder_treatment.numpy())


def read_data(database='SHHS1'):
    data_df = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                       'data_' + database + '.csv'))
    spo2_features = data_df.drop(columns=x_columns + y_column + t_baseline + event_columns)
    spo2_features = preprocess_data(spo2_features)
    other_data = data_df[x_columns + y_column + t_baseline]

    all_data = pd.concat([spo2_features, other_data], axis=1)
    all_data = all_data.dropna()

    spo2_features = all_data.drop(columns=x_columns + y_column + t_baseline)
    other_data = all_data[x_columns + y_column + t_baseline]

    return spo2_features, other_data


def main(database):
    spo2_features, other_data = read_data(database)

    autoencoder_treatment = autoencoder_reduction(spo2_features)
    other_data['autoencoder_treatment'] = autoencoder_treatment

    pca_treatment = pca_reduction(spo2_features)
    other_data['pca_treatment'] = pca_treatment

    other_data.to_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                   'data_' + database + '_dimensionality_reduction.csv'), index=False)


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')

    # main(database='SHHS1')
    main(database='SHHS2')
