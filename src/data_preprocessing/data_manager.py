import os.path
import pickle
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from src.data_preprocessing.paths import *


class DataManager:
    FREQUENCIES = {'respiban_ecg': 700, 'respiban_emg': 700, 'respiban_eda': 700, 'respiban_temperature': 700,
                   'respiban_respiration': 700, 'respiban_x': 700, 'respiban_y': 700, 'respiban_z': 700,
                   'empatica_bvp': 64, 'empatica_eda': 4, 'empatica_temperature': 4, 'empatica_x': 32, 'empatica_y': 32,
                   'empatica_z': 32, 'label': 700}

    def load_data(self, hz):
        data = []
        for d in os.listdir(WESAD_DIRECTORY):
            if re.match('S[0-9]', d):
                current_s = os.path.join(WESAD_DIRECTORY, d)
                current_pkl = os.path.join(current_s, d + '.pkl')
                dictionary = self._load_subject_data(current_pkl)
                dictionary = self._extract_accelerometer(dictionary)
                dictionary = self._reformat_dictionary(dictionary)
                dictionary = self._resample_data(dictionary, hz)
                dictionary = self._normalize(dictionary)
                data.append(dictionary)
        return data

    def _load_subject_data(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            return pickle.load(file)

    def _extract_accelerometer(self, data):
        # extract respiban accelerometer entries
        respiban_acc = data['signal']['chest']['ACC']
        respiban_x = np.array([[accelerometer[0]] for accelerometer in respiban_acc])
        respiban_y = np.array([[accelerometer[1]] for accelerometer in respiban_acc])
        respiban_z = np.array([[accelerometer[2]] for accelerometer in respiban_acc])
        data['signal']['chest'] |= {'respiban_x': respiban_x}
        data['signal']['chest'] |= {'respiban_y': respiban_y}
        data['signal']['chest'] |= {'respiban_z': respiban_z}

        # extract empatica accelerometer entries
        empatica_acc = data['signal']['wrist']['ACC']
        empatica_x = np.array([[accelerometer[0]] for accelerometer in empatica_acc])
        empatica_y = np.array([[accelerometer[1]] for accelerometer in empatica_acc])
        empatica_z = np.array([[accelerometer[2]] for accelerometer in empatica_acc])
        data['signal']['wrist'] |= {'empatica_x': empatica_x}
        data['signal']['wrist'] |= {'empatica_y': empatica_y}
        data['signal']['wrist'] |= {'empatica_z': empatica_z}

        return data

    def split_data(self):
        pass

    def _reformat_dictionary(self, data):
        new_data = {'label': data['label']}
        new_data |= {'respiban_x': data['signal']['chest']['respiban_x']}
        new_data |= {'respiban_y': data['signal']['chest']['respiban_y']}
        new_data |= {'respiban_z': data['signal']['chest']['respiban_z']}
        new_data |= {'respiban_ecg': data['signal']['chest']['ECG']}
        new_data |= {'respiban_emg': data['signal']['chest']['EMG']}
        new_data |= {'respiban_eda': data['signal']['chest']['EDA']}
        new_data |= {'respiban_temperature': data['signal']['chest']['Temp']}
        new_data |= {'respiban_respiration': data['signal']['chest']['Resp']}
        new_data |= {'empatica_x': data['signal']['wrist']['empatica_x']}
        new_data |= {'empatica_y': data['signal']['wrist']['empatica_y']}
        new_data |= {'empatica_z': data['signal']['wrist']['empatica_z']}
        new_data |= {'empatica_bvp': data['signal']['wrist']['BVP']}
        new_data |= {'empatica_eda': data['signal']['wrist']['EDA']}
        new_data |= {'empatica_temperature': data['signal']['wrist']['TEMP']}
        return new_data

    def _resample_data(self, data, hz):
        # reshape all arrays with their respective original frequencies
        for k, v in data.items():
            new_shape = (int(len(v) / self.FREQUENCIES[k]), self.FREQUENCIES[k])
            data.update({k: np.reshape(v, new_shape)})
        # resampling all signals
        for k, v in data.items():
            resampled_array = np.empty((0,))
            for item in v:
                resampled_array = np.append(resampled_array,
                                            np.interp(np.linspace(0, len(item) - 1, hz), np.arange(len(item)), item))
            data.update({k: resampled_array})
        return data

    def _split_array_by_percentage(self, array, percentage):
        split_index = int((percentage / 100) * array.shape[0])
        return np.split(array, [split_index])

    def _normalize(self, data):
        scaler = MinMaxScaler()
        for k, v in data.items():
            if k != 'label':
                normalized_signal = scaler.fit_transform(v.reshape(-1, 1))
                data.update({k: normalized_signal})
        return data
