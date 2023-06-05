import os.path
import pickle
import numpy as np
import re
from paths import *


class DataManager:

    def load_data(self):
        for d in os.listdir(WESAD_DIRECTORY):
            if re.match('S[0-9]', d):
                current_S = os.path.join(WESAD_DIRECTORY, d)
                print(current_S)
                current_pkl = os.path.join(current_S, d + '.pkl')
                data = self._load_subject_data(current_pkl)
                data = self._extract_accelerometer(data)
                data = self._reformat_dictionary(data)
                data = self._resample_data(data)
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

        # delete acc entries on the dictionary
        del data['signal']['chest']['ACC']
        del data['signal']['wrist']['ACC']
        return data

    def split_data(self):

        data

    def _reformat_dictionary(self, data):
        labels = data['label']
        # extract all signals from sub-dictionaries
        data = {k: v for nested_dict in data['signal'].values() for k, v in
                nested_dict.items()}

        # group all signals sub-arrays into one single array
        data = {k: np.array([x[0] for x in v]) for k, v in data.items()}

        # add back labels
        data |= {'label': labels}
        return data

    def _resample_data(self, data):
        # resampling all signals at 700Hz (4255300 elements)
        for k, v in data.items():
            data.update({k: np.interp(np.linspace(0, len(v) - 1, 4255300), np.arange(len(v)), v)})
        return data

    def _split_array_by_percentage(self, array, percentage):
        split_index = int((percentage / 100) * array.shape[0])
        return np.split(array, [split_index])


manager = DataManager()
data = manager.load_data()
respiration = data['Resp']
print(len(respiration))
print(len(manager._split_array_by_percentage(respiration, 70)[0]))
