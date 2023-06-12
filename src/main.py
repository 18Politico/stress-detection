import sys
import os
import pickle
import pathlib
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal

sys.path.append(os.path.join('data_preprocessing'))
from data_preprocessing.data_manager import DataManager
from data_preprocessing.paths import *
import matplotlib.pyplot as plt

# 32 Hz 175872
# 700 Hz 4255300

with open(RAW_DATA_PKL, 'rb') as f:
    manager = DataManager()
    data = pickle.load(f)
    data = manager.preprocess_raw_data(data)
    data_cross = data[0]
    del data[0]
    # MODEL DEFINITION
    # Define the neural network architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(48,)))
    # Dropout layer with a dropout rate of 0.5
    model.add(tf.keras.layers.Dropout(.3))
    # Add the second hidden layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # Dropout layer with a dropout rate of 0.3
    model.add(tf.keras.layers.Dropout(.3))
    # Add the output layer for classification
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
    # MODEL DEFINITION

    # DATA CROSS
    feature_1_cross = tf.constant(data_cross['respiban_ecg_mean'], dtype=tf.float32)
    feature_2_cross = tf.constant(data_cross['respiban_ecg_variance'], dtype=tf.float32)
    feature_3_cross = tf.constant(data_cross['respiban_ecg_std'], dtype=tf.float32)
    feature_4_cross = tf.constant(data_cross['respiban_ecg_dynamic_range'], dtype=tf.float32)
    feature_5_cross = tf.constant(data_cross['respiban_ecg_max'], dtype=tf.float32)
    feature_6_cross = tf.constant(data_cross['respiban_ecg_min'], dtype=tf.float32)
    feature_7_cross = tf.constant(data_cross['respiban_emg_mean'], dtype=tf.float32)
    feature_8_cross = tf.constant(data_cross['respiban_emg_variance'], dtype=tf.float32)
    feature_9_cross = tf.constant(data_cross['respiban_emg_std'], dtype=tf.float32)
    feature_10_cross = tf.constant(data_cross['respiban_emg_dynamic_range'], dtype=tf.float32)
    feature_11_cross = tf.constant(data_cross['respiban_emg_max'], dtype=tf.float32)
    feature_12_cross = tf.constant(data_cross['respiban_emg_min'], dtype=tf.float32)
    feature_13_cross = tf.constant(data_cross['respiban_eda_mean'], dtype=tf.float32)
    feature_14_cross = tf.constant(data_cross['respiban_eda_variance'], dtype=tf.float32)
    feature_15_cross = tf.constant(data_cross['respiban_eda_std'], dtype=tf.float32)
    feature_16_cross = tf.constant(data_cross['respiban_eda_dynamic_range'], dtype=tf.float32)
    feature_17_cross = tf.constant(data_cross['respiban_eda_max'], dtype=tf.float32)
    feature_18_cross = tf.constant(data_cross['respiban_eda_min'], dtype=tf.float32)
    feature_19_cross = tf.constant(data_cross['respiban_temperature_mean'], dtype=tf.float32)
    feature_20_cross = tf.constant(data_cross['respiban_temperature_variance'], dtype=tf.float32)
    feature_21_cross = tf.constant(data_cross['respiban_temperature_std'], dtype=tf.float32)
    feature_22_cross = tf.constant(data_cross['respiban_temperature_dynamic_range'], dtype=tf.float32)
    feature_23_cross = tf.constant(data_cross['respiban_temperature_max'], dtype=tf.float32)
    feature_24_cross = tf.constant(data_cross['respiban_temperature_min'], dtype=tf.float32)
    feature_25_cross = tf.constant(data_cross['respiban_respiration_mean'], dtype=tf.float32)
    feature_26_cross = tf.constant(data_cross['respiban_respiration_variance'], dtype=tf.float32)
    feature_27_cross = tf.constant(data_cross['respiban_respiration_std'], dtype=tf.float32)
    feature_28_cross = tf.constant(data_cross['respiban_respiration_dynamic_range'], dtype=tf.float32)
    feature_29_cross = tf.constant(data_cross['respiban_respiration_max'], dtype=tf.float32)
    feature_30_cross = tf.constant(data_cross['respiban_respiration_min'], dtype=tf.float32)
    feature_31_cross = tf.constant(data_cross['respiban_x_mean'], dtype=tf.float32)
    feature_32_cross = tf.constant(data_cross['respiban_x_variance'], dtype=tf.float32)
    feature_33_cross = tf.constant(data_cross['respiban_x_std'], dtype=tf.float32)
    feature_34_cross = tf.constant(data_cross['respiban_x_dynamic_range'], dtype=tf.float32)
    feature_35_cross = tf.constant(data_cross['respiban_x_max'], dtype=tf.float32)
    feature_36_cross = tf.constant(data_cross['respiban_x_min'], dtype=tf.float32)
    feature_37_cross = tf.constant(data_cross['respiban_y_mean'], dtype=tf.float32)
    feature_38_cross = tf.constant(data_cross['respiban_y_variance'], dtype=tf.float32)
    feature_39_cross = tf.constant(data_cross['respiban_y_std'], dtype=tf.float32)
    feature_40_cross = tf.constant(data_cross['respiban_y_dynamic_range'], dtype=tf.float32)
    feature_41_cross = tf.constant(data_cross['respiban_y_max'], dtype=tf.float32)
    feature_42_cross = tf.constant(data_cross['respiban_y_min'], dtype=tf.float32)
    feature_43_cross = tf.constant(data_cross['respiban_z_mean'], dtype=tf.float32)
    feature_44_cross = tf.constant(data_cross['respiban_z_variance'], dtype=tf.float32)
    feature_45_cross = tf.constant(data_cross['respiban_z_std'], dtype=tf.float32)
    feature_46_cross = tf.constant(data_cross['respiban_z_dynamic_range'], dtype=tf.float32)
    feature_47_cross = tf.constant(data_cross['respiban_z_max'], dtype=tf.float32)
    feature_48_cross = tf.constant(data_cross['respiban_z_min'], dtype=tf.float32)

    labels_cross = tf.constant(data_cross['label'] - 1, dtype=tf.float32)

    # Combine features into a single tensor
    features_cross = tf.stack(
        [feature_1_cross, feature_2_cross, feature_3_cross, feature_4_cross, feature_5_cross, feature_6_cross,
         feature_7_cross, feature_8_cross, feature_9_cross,
         feature_10_cross,
         feature_11_cross, feature_12_cross, feature_13_cross, feature_14_cross, feature_15_cross, feature_16_cross,
         feature_17_cross, feature_18_cross, feature_19_cross,
         feature_20_cross, feature_21_cross, feature_22_cross,
         feature_23_cross, feature_24_cross, feature_25_cross, feature_26_cross, feature_27_cross, feature_28_cross,
         feature_29_cross, feature_30_cross, feature_31_cross,
         feature_32_cross, feature_33_cross, feature_34_cross, feature_35_cross,
         feature_36_cross, feature_37_cross, feature_38_cross, feature_39_cross, feature_40_cross, feature_41_cross,
         feature_42_cross, feature_43_cross, feature_44_cross,
         feature_45_cross, feature_46_cross, feature_47_cross, feature_48_cross], axis=1)

    # split data into train and test
    train_features_cross, test_features_cross, train_labels_cross, test_labels_cross = train_test_split(
        np.array(features_cross), np.array(labels_cross), test_size=0.2, random_state=42)

    # DATA CROSS
    for ele in data:
        data = ele
        feature_1 = tf.constant(data['respiban_ecg_mean'], dtype=tf.float32)
        feature_2 = tf.constant(data['respiban_ecg_variance'], dtype=tf.float32)
        feature_3 = tf.constant(data['respiban_ecg_std'], dtype=tf.float32)
        feature_4 = tf.constant(data['respiban_ecg_dynamic_range'], dtype=tf.float32)
        feature_5 = tf.constant(data['respiban_ecg_max'], dtype=tf.float32)
        feature_6 = tf.constant(data['respiban_ecg_min'], dtype=tf.float32)
        feature_7 = tf.constant(data['respiban_emg_mean'], dtype=tf.float32)
        feature_8 = tf.constant(data['respiban_emg_variance'], dtype=tf.float32)
        feature_9 = tf.constant(data['respiban_emg_std'], dtype=tf.float32)
        feature_10 = tf.constant(data['respiban_emg_dynamic_range'], dtype=tf.float32)
        feature_11 = tf.constant(data['respiban_emg_max'], dtype=tf.float32)
        feature_12 = tf.constant(data['respiban_emg_min'], dtype=tf.float32)
        feature_13 = tf.constant(data['respiban_eda_mean'], dtype=tf.float32)
        feature_14 = tf.constant(data['respiban_eda_variance'], dtype=tf.float32)
        feature_15 = tf.constant(data['respiban_eda_std'], dtype=tf.float32)
        feature_16 = tf.constant(data['respiban_eda_dynamic_range'], dtype=tf.float32)
        feature_17 = tf.constant(data['respiban_eda_max'], dtype=tf.float32)
        feature_18 = tf.constant(data['respiban_eda_min'], dtype=tf.float32)
        feature_19 = tf.constant(data['respiban_temperature_mean'], dtype=tf.float32)
        feature_20 = tf.constant(data['respiban_temperature_variance'], dtype=tf.float32)
        feature_21 = tf.constant(data['respiban_temperature_std'], dtype=tf.float32)
        feature_22 = tf.constant(data['respiban_temperature_dynamic_range'], dtype=tf.float32)
        feature_23 = tf.constant(data['respiban_temperature_max'], dtype=tf.float32)
        feature_24 = tf.constant(data['respiban_temperature_min'], dtype=tf.float32)
        feature_25 = tf.constant(data['respiban_respiration_mean'], dtype=tf.float32)
        feature_26 = tf.constant(data['respiban_respiration_variance'], dtype=tf.float32)
        feature_27 = tf.constant(data['respiban_respiration_std'], dtype=tf.float32)
        feature_28 = tf.constant(data['respiban_respiration_dynamic_range'], dtype=tf.float32)
        feature_29 = tf.constant(data['respiban_respiration_max'], dtype=tf.float32)
        feature_30 = tf.constant(data['respiban_respiration_min'], dtype=tf.float32)
        feature_31 = tf.constant(data['respiban_x_mean'], dtype=tf.float32)
        feature_32 = tf.constant(data['respiban_x_variance'], dtype=tf.float32)
        feature_33 = tf.constant(data['respiban_x_std'], dtype=tf.float32)
        feature_34 = tf.constant(data['respiban_x_dynamic_range'], dtype=tf.float32)
        feature_35 = tf.constant(data['respiban_x_max'], dtype=tf.float32)
        feature_36 = tf.constant(data['respiban_x_min'], dtype=tf.float32)
        feature_37 = tf.constant(data['respiban_y_mean'], dtype=tf.float32)
        feature_38 = tf.constant(data['respiban_y_variance'], dtype=tf.float32)
        feature_39 = tf.constant(data['respiban_y_std'], dtype=tf.float32)
        feature_40 = tf.constant(data['respiban_y_dynamic_range'], dtype=tf.float32)
        feature_41 = tf.constant(data['respiban_y_max'], dtype=tf.float32)
        feature_42 = tf.constant(data['respiban_y_min'], dtype=tf.float32)
        feature_43 = tf.constant(data['respiban_z_mean'], dtype=tf.float32)
        feature_44 = tf.constant(data['respiban_z_variance'], dtype=tf.float32)
        feature_45 = tf.constant(data['respiban_z_std'], dtype=tf.float32)
        feature_46 = tf.constant(data['respiban_z_dynamic_range'], dtype=tf.float32)
        feature_47 = tf.constant(data['respiban_z_max'], dtype=tf.float32)
        feature_48 = tf.constant(data['respiban_z_min'], dtype=tf.float32)
        '''feature_9 = data['empatica_bvp']
        feature_10 = data['empatica_temperature']
        feature_11 = data['empatica_x']
        feature_12 = data['empatica_y']
        feature_13 = data['empatica_z']
        feature_14 = data['empatica_eda']'''
        labels = tf.constant(data['label'] - 1, dtype=tf.float32)

        # Combine features into a single tensor
        features = tf.stack(
            [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9,
             feature_10,
             feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19,
             feature_20, feature_21, feature_22,
             feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31,
             feature_32, feature_33, feature_34, feature_35,
             feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44,
             feature_45, feature_46, feature_47, feature_48], axis=1)

        # split data into train and test
        train_features, test_features, train_labels, test_labels = train_test_split(
            np.array(features), np.array(labels), test_size=0.2, random_state=42)

        model.fit(train_features, train_labels, epochs=50, batch_size=32,
                  validation_data=(test_features_cross, test_labels_cross),
                  callbacks=[early_stopping])

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_features_cross, test_labels_cross)
    print('Test loss:', loss)
    print("Test Accuracy:", accuracy)
