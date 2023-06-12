import os.path
import pickle
import numpy as np
import re

import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from paths import *
from scipy import signal


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

    # Label values defined in the WESAD readme
    BASELINE = 1
    STRESS = 2
    AMUSEMENT = 3
    MEDITATION = 4

    def load_raw_data(self):
        data = []
        for d in os.listdir(WESAD_DIRECTORY):
            if re.match('S[0-9]', d):
                current_pkl = os.path.join(WESAD_DIRECTORY, d, d + '.pkl')
                dictionary = self._load_subject_data(current_pkl)
                data.append(dictionary)
        return data

    def extract_signals_from_raw_data(self, data):
        self._reformat_raw_data(data)
        return data

    def preprocess_raw_data(self, data):
        for dictionary in data:
            # extract subject
            self._resample(dictionary)
            self._drop_unnecessary_records(dictionary)
            self._normalize(dictionary)
            self._split_data(dictionary)
            self._extract_features(dictionary)
            dictionary['label'] = np.array(
                [lbl[-1] for lbl in dictionary['label'].reshape(int(len(dictionary['label']) / 64), 64)]).reshape(-1)
        return data

    def prepare_to_preprocessing(self, data):
        self._reformat_raw_data(data)
        return self._merge_dictionaries(data)

    def _upsample(self, signals, original_hz, target_hz):
        padding_length = target_hz - original_hz
        padding = [0] * padding_length
        upsampled_signals = []
        for s in signals:
            upsampled_signals.append(list(s) + padding)

        '''upsampled_signals = []
        padding_length = target_hz - original_hz
        padding_per_value = padding_length // original_hz
        remaining_padding = padding_length % original_hz

        for signal in signals:
            upsampled_signal = []
            for value in signal:
                upsampled_signal.append(value)
                upsampled_signal.extend([0] * padding_per_value)
                if remaining_padding > 0:
                    upsampled_signal.append(0)
                    remaining_padding -= 1
            upsampled_signals.append(upsampled_signal)'''

        return np.array(upsampled_signals).reshape(-1)

    ''' 
     for i, dictionary in enumerate(data):
         dictionary = self._reformat_data(dictionary)
         dictionary = self._resample(dictionary, 32)
         dictionary = self._drop_unnecessary_records(dictionary)
         dictionary = self._normalize(dictionary)
         data[i] = dictionary
     return self._merge_dictionaries(data)'''

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

    def _reformat_raw_data(self, data):
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

    def _resample(self, dictionary):
        # downsampling labels
        dictionary.update({'label': self._downsampling_labels(dictionary['label'])})
        # resample
        for k, v in dictionary.items():
            if k != 'label' and 'subject':
                # reshape all signals with their respective original frequencies
                new_shape = (int(len(v) / self.FREQUENCIES[k]), self.FREQUENCIES[k])
                v = np.reshape(v, new_shape)
                if k in self.EMPATICA_SIGNALS:
                    v = self._upsample(v, self.FREQUENCIES[k], 64)
                elif k in self.RESIBAN_SIGNALS:
                    v = self._downsample(v, self.FREQUENCIES[k], 64)
                dictionary.update({k: v})
        return dictionary

    '''for k, v in dictionary.items():
            # reshape all signals with their respective original frequencies
            new_shape = (int(len(v) / self.FREQUENCIES[k]), self.FREQUENCIES[k])
            v = np.reshape(v, new_shape)
            # resampling the signal
            v = self._upsample(v, self.FREQUENCIES[k], 32)
            dictionary.update({k: np.array(v).reshape(-1)})
        dictionary |= {'label': labels}
        return dictionary'''

    def _downsampling_labels(self, labels):
        return np.array([lbl[:64] for lbl in labels.reshape(int(len(labels) / 700), 700)]).reshape(-1)

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

    def _sliding_window(self, arr, window_size, shift):
        # features we want to compute
        mean_tmp = []
        std_tmp = []
        dynamic_range_tmp = []
        max_tmp = []
        min_tmp = []
        variance_tmp = []

        # sliding window
        start = 0
        end = start + window_size
        while end <= len(arr):
            mean_tmp.append(np.mean(arr[start:end]))
            variance_tmp.append(np.var(arr[start:end]))
            std_tmp.append(np.sqrt(variance_tmp[-1]))
            dynamic_range_tmp.append(np.ptp(arr[start:end]))
            max_tmp.append(np.amax(arr[start:end]))
            min_tmp.append((np.amin(arr[start:end])))
            end += window_size
            start += shift
        return mean_tmp, variance_tmp, std_tmp, dynamic_range_tmp, max_tmp, min_tmp

    def _extract_features(self, dictionary):
        for s in self.RESIBAN_SIGNALS + self.EMPATICA_SIGNALS:
            features = self._sliding_window(dictionary[s], 64, 64)
            dictionary[f'{s}_mean'] = features[0]
            dictionary[f'{s}_variance'] = features[1]
            dictionary[f'{s}_std'] = features[2]
            dictionary[f'{s}_dynamic_range'] = features[3]
            dictionary[f'{s}_max'] = features[4]
            dictionary[f'{s}_min'] = features[5]
            del dictionary[s]
        return dictionary

    def _split_data(self, dictionary):
        dictionary.update()
        return dictionary
