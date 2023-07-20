import os.path
import pickle
import numpy as np
import tensorflow as tf
import random
import re
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from scipy import signal

ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.path.abspath(__file__)))))
WESAD_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'dataset', 'WESAD')
RAW_DATA_PKL = os.path.join(ROOT_DIRECTORY, 'dataset', 'raw_data.pkl')
PREPROCESSED_DATA = os.path.join(ROOT_DIRECTORY, 'dataset', 'preprocessed_data.pkl')


class DataManager:
    FREQUENCIES = {'respiban_ecg': 700, 'respiban_emg': 700, 'respiban_eda': 700, 'respiban_temperature': 700,
                   'respiban_respiration': 700, 'respiban_x': 700, 'respiban_y': 700, 'respiban_z': 700,
                   'empatica_bvp': 64, 'empatica_eda': 4, 'empatica_temperature': 4, 'empatica_x': 32, 'empatica_y': 32,
                   'empatica_z': 32, 'label': 700}

    DICTIONARY_KEYS = ['respiban_x', 'respiban_y', 'respiban_z', 'respiban_ecg', 'respiban_emg', 'respiban_eda',
                       'respiban_temperature', 'respiban_respiration', 'empatica_x', 'empatica_y', 'empatica_z',
                       'empatica_bvp',
                       'empatica_eda', 'empatica_temperature', 'label']

    RESIBAN_SIGNALS = ['respiban_x', 'respiban_y', 'respiban_z', 'respiban_ecg', 'respiban_emg', 'respiban_eda',
                       'respiban_temperature', 'respiban_respiration']

    EMPATICA_SIGNALS = ['empatica_x', 'empatica_y', 'empatica_z', 'empatica_bvp',
                        'empatica_eda', 'empatica_temperature']

    RESAMPLING_RATE = 64
    # in seconds
    WINDOW_SIZE = 5

    # Label values defined in the WESAD readme
    BASELINE = 1
    STRESS = 2
    AMUSEMENT = 3
    MEDITATION = 4

    # Label values in our binary case
    BINARY_NO_STRESS = 0
    BINARY_STRESS = 1

    LEARNING_RATE = 0.001
    L2_PENALTY = 0.001
    BATCH_SIZE = 32
    EPOCHS = 50
    SEED_VALUE = 42

    def set_random_seed(self):
        # Set seed values
        np.random.seed(self.SEED_VALUE)
        tf.random.set_seed(self.SEED_VALUE)
        random.seed(self.SEED_VALUE)

    def load_raw_data(self):
        data = []
        for d in os.listdir(WESAD_DIRECTORY):
            if re.match('S[0-9]', d):
                current_pkl = os.path.join(WESAD_DIRECTORY, d, d + '.pkl')
                dictionary = self._load_subject_data(current_pkl)
                data.append(dictionary)
        return data

    def reformat_raw_data(self, data):
        for i, dictionary in enumerate(data):
            self._extract_accelerometer(dictionary)
            new_dictionary = {'label': dictionary['label'].reshape(-1)}
            new_dictionary |= {'respiban_x': dictionary['signal']['chest']['respiban_x'].reshape(-1)}
            new_dictionary |= {'respiban_y': dictionary['signal']['chest']['respiban_y'].reshape(-1)}
            new_dictionary |= {'respiban_z': dictionary['signal']['chest']['respiban_z'].reshape(-1)}
            new_dictionary |= {'respiban_ecg': dictionary['signal']['chest']['ECG'].reshape(-1)}
            new_dictionary |= {'respiban_emg': dictionary['signal']['chest']['EMG'].reshape(-1)}
            new_dictionary |= {'respiban_eda': dictionary['signal']['chest']['EDA'].reshape(-1)}
            new_dictionary |= {'respiban_temperature': dictionary['signal']['chest']['Temp'].reshape(-1)}
            new_dictionary |= {'respiban_respiration': dictionary['signal']['chest']['Resp'].reshape(-1)}
            new_dictionary |= {'empatica_x': dictionary['signal']['wrist']['empatica_x'].reshape(-1)}
            new_dictionary |= {'empatica_y': dictionary['signal']['wrist']['empatica_y'].reshape(-1)}
            new_dictionary |= {'empatica_z': dictionary['signal']['wrist']['empatica_z'].reshape(-1)}
            new_dictionary |= {'empatica_bvp': dictionary['signal']['wrist']['BVP'].reshape(-1)}
            new_dictionary |= {'empatica_eda': dictionary['signal']['wrist']['EDA'].reshape(-1)}
            new_dictionary |= {'empatica_temperature': dictionary['signal']['wrist']['TEMP'].reshape(-1)}
            new_dictionary |= {'subject': dictionary['subject']}
            data[i] = new_dictionary
        return data

    def preprocess_raw_data(self, data):
        for dictionary in data:
            self._resample(dictionary)
            self._drop_unnecessary_records(dictionary)
            self._normalize(dictionary)
            self._map_labels_to_binary_case(dictionary)
        return data

    def prepare_data_for_cnn(self, data):
        x, y = None, None
        for dictionary in data:
            subject = dictionary['subject']
            del dictionary['subject']
            # split signals into stress and no stress
            stress_indexes = np.nonzero(np.isin(dictionary['label'], 1))
            no_stress_indexes = np.nonzero(np.isin(dictionary['label'], 0))
            stress_dictionary = self._retrieve_indexes(dictionary, stress_indexes)
            no_stress_dictionary = self._retrieve_indexes(dictionary, no_stress_indexes)
            # compute spectograms
            x_stress, y_stress = self._compute_spectogram(stress_dictionary, self.WINDOW_SIZE * 64)
            x_no_stress, y_no_stress = self._compute_spectogram(no_stress_dictionary, self.WINDOW_SIZE * 64)
            # merge stress and no_stress data
            x_to_add = np.concatenate((x_stress, x_no_stress), axis=0)
            y_to_add = np.concatenate((y_stress, y_no_stress), axis=0)
            # add data of the subject
            if x is None:
                x = x_to_add
                y = y_to_add
            else:
                x = np.concatenate((x, x_to_add), axis=0)
                y = np.concatenate((y, y_to_add), axis=0)
            # add back the subject
            dictionary['subject'] = subject
        # Shuffle the data
        shuffled_indices = np.random.permutation(len(x))
        x = x[shuffled_indices]
        y = y[shuffled_indices]
        # Normalize the spectrograms
        x = (x - np.mean(x)) / np.std(x)
        return x, y

    def train_cnn(self, x, y, x_validation, y_validation, input_shape):
        # Define the CNN model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(self.L2_PENALTY)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Define early stopping callback to restore weights from the epoch with the best validation loss
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5,
                                       restore_best_weights=True)

        # Train the model
        model.fit(x, y, validation_data=(x_validation, y_validation), batch_size=self.BATCH_SIZE, epochs=self.EPOCHS,
                  callbacks=[early_stopping])
        return model.evaluate(x_validation, y_validation)
      
