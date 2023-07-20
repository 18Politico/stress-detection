import os.path
import pickle
import numpy as np
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

    def _retrieve_indexes(self, dictionary, mask):
        result = {}
        for key, array in dictionary.items():
            result[key] = dictionary[key][mask]
        return result

    def _upsample(self, signals, original_hz, target_hz):
        padding_length = target_hz - original_hz
        padding = [0] * padding_length
        upsampled_signals = []
        for s in signals:
            upsampled_signals.append(list(s) + padding)
        return np.array(upsampled_signals).reshape(-1)

    def _load_subject_data(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            return pickle.load(file)

    def _extract_accelerometer(self, dictionary):
        # extract respiban accelerometer entries
        respiban_acc = dictionary['signal']['chest']['ACC']
        respiban_x = np.array([[accelerometer[0]] for accelerometer in respiban_acc])
        respiban_y = np.array([[accelerometer[1]] for accelerometer in respiban_acc])
        respiban_z = np.array([[accelerometer[2]] for accelerometer in respiban_acc])
        dictionary['signal']['chest'] |= {'respiban_x': respiban_x}
        dictionary['signal']['chest'] |= {'respiban_y': respiban_y}
        dictionary['signal']['chest'] |= {'respiban_z': respiban_z}
        del dictionary['signal']['chest']['ACC']
        # extract empatica accelerometer entries
        empatica_acc = dictionary['signal']['wrist']['ACC']
        empatica_x = np.array([[accelerometer[0]] for accelerometer in empatica_acc])
        empatica_y = np.array([[accelerometer[1]] for accelerometer in empatica_acc])
        empatica_z = np.array([[accelerometer[2]] for accelerometer in empatica_acc])
        dictionary['signal']['wrist'] |= {'empatica_x': empatica_x}
        dictionary['signal']['wrist'] |= {'empatica_y': empatica_y}
        dictionary['signal']['wrist'] |= {'empatica_z': empatica_z}
        del dictionary['signal']['wrist']['ACC']

    def _resample(self, dictionary):
        # downsampling labels
        dictionary.update({'label': self._downsampling_labels(dictionary['label'])})
        # resample
        for k, v in dictionary.items():
            if k != 'label' and k != 'subject':
                # reshape all signals with their respective original frequencies
                new_shape = (int(len(v) / self.FREQUENCIES[k]), self.FREQUENCIES[k])
                v = np.reshape(v, new_shape)
                if k in self.EMPATICA_SIGNALS:
                    v = self._upsample(v, self.FREQUENCIES[k], self.RESAMPLING_RATE)
                elif k in self.RESIBAN_SIGNALS:
                    v = self._downsample(v, self.FREQUENCIES[k], self.RESAMPLING_RATE)
                dictionary.update({k: v})
        return dictionary

    def _downsampling_labels(self, labels):
        return np.array([lbl[:self.RESAMPLING_RATE] for lbl in labels.reshape(int(len(labels) / 700), 700)]).reshape(-1)

    def _normalize(self, dictionary):
        scaler = MinMaxScaler()
        for k, v in dictionary.items():
            # skip label normalization
            if k != 'label' and k != 'subject':
                # normalize in a range between 0 and 1 all signals
                normalized_signal = scaler.fit_transform(v.reshape(-1, 1))
                dictionary.update({k: normalized_signal.reshape(-1)})

    def _drop_unnecessary_records(self, dictionary):
        useful_indexes = np.nonzero(np.isin(dictionary['label'], [1, 2, 3, 4]))
        for k, v in dictionary.items():
            if k != 'subject':
                dictionary.update({k: v[useful_indexes]})

    def _merge_dictionaries(self, data):
        merged_dict = {key: np.empty((0,)) for key in self.DICTIONARY_KEYS}
        for dictionary in data:
            for key, value in dictionary.items():
                merged_dict[key] = np.concatenate((merged_dict[key], value))
        return merged_dict

    def _downsample(self, signals, original_hz, target_hz):
        scaling_factor = original_hz / target_hz
        downsampled_signals = []
        for s in signals:
            downsampled_signals.append(signal.decimate(s, round(scaling_factor)))
        return np.array(downsampled_signals).reshape(-1)

    def _split_data(self, dictionary, stress_data, no_stress_data):
        stress_indexes = np.nonzero(np.isin(dictionary['label'], [1]))
        no_stress_indexes = np.nonzero(np.isin(dictionary['label'], [0]))
        stress_dict = dict()
        no_stress_dict = dict()

        for s in self.DICTIONARY_KEYS:
            stress_dict[s] = dictionary[s][stress_indexes]
            no_stress_dict[s] = dictionary[s][no_stress_indexes]
            stress_data |= stress_dict
            no_stress_data |= no_stress_dict
        return stress_data, no_stress_data

    def _map_labels_to_binary_case(self, dictionary):
        mapping = {1: self.BINARY_NO_STRESS, 2: self.BINARY_STRESS, 3: self.BINARY_NO_STRESS, 4: self.BINARY_NO_STRESS}
        dictionary['label'] = np.array([mapping.get(label, label) for label in dictionary['label']])
        return dictionary

