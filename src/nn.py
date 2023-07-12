import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# 'signals' is the new 'chest'

bound = 18

excluded = 17

def regroup_data(wesad_dict: dict) -> dict:
    new_wesad_dict = dict()
    for subject in range(2, bound):
        if subject == 12:
            continue
        subject_key = 'S' + str(subject)
        new_wesad_dict[subject_key] = subject_splitted_dict(wesad_dict, subject_key)
    return new_wesad_dict


def subject_splitted_dict(wesad_dict: dict, subject_key: str):
    nested_subject_dict = wesad_dict[subject_key]
    subject_labels = nested_subject_dict['labels']
    new_nested_wesad_dict = dict()
    for output in range(0, 4):
        output_indexes = np.nonzero(np.isin(subject_labels, output))
        output_dict = dict()
        output_labels = subject_labels[output_indexes]
        for k, v in dict(nested_subject_dict['chest']).items():
            output_data = v[output_indexes]
            output_data = make_array_size_multiple_of(700, output_data)
            output_dict[k] = output_data
        output_key = str(output)
        tmp = dict()
        tmp['signals'] = output_dict
        tmp['labels'] = make_array_size_multiple_of(700, output_labels)
        new_nested_wesad_dict[output_key] = tmp
    return new_nested_wesad_dict


def make_array_size_multiple_of(number: int, array: []):
    new_size = len(array) // number * number
    trimmed_array = array[:new_size]
    return trimmed_array


def get_slicing_windows_wesad(new_wesad: dict) -> dict:
    slicing_windows_wesad = dict()
    for subject in range(2, bound):  # FINO A 17
        if subject == 12:
            continue
        subject_key = 'S' + str(subject)
        slicing_windows_wesad[subject_key] = populate_slicing_window_dict(new_wesad, subject_key)
    return slicing_windows_wesad


def populate_slicing_window_dict(new_wesad: dict, subject_key: str) -> dict:
    new_sliding_windows_subject_dict = dict()
    subject_dict = dict(new_wesad[subject_key])
    for output_key, output_dict in subject_dict.items():
        data_dict = dict()
        for name_data, data in dict(output_dict['signals']).items():
            data_stats_dict = calculate_windows_stats_on(data)
            data_dict[name_data] = data_stats_dict
        tmp = dict()
        tmp['signals'] = data_dict
        needed_labels_len = int(len(np.array(output_dict['labels']))/700)
        tmp['labels'] = output_dict['labels'][:needed_labels_len]
        new_sliding_windows_subject_dict[output_key] = tmp
    return new_sliding_windows_subject_dict


def calculate_windows_stats_on(data: []) -> dict:
    data_stats_dict = dict()
    means = []
    STDs = []
    MAXs = []
    MINs = []
    variances = []
    for i in range(0, len(data), 700):
        window = data[i:i + 700]
        mean = np.mean(window)
        means.append(mean)
        STD = np.std(window)
        STDs.append(STD)
        MAX = np.max(window)
        MAXs.append(MAX)
        MIN = np.min(window)
        MINs.append(MIN)
        variance = np.var(window)
        variances.append(variance)
        # dynamic_range !!!!!!!!!!!!!!
    data_stats_dict['means'] = np.array(means)
    data_stats_dict['STDs'] = np.array(STDs)
    data_stats_dict['MAXs'] = np.array(MAXs)
    data_stats_dict['MINs'] = np.array(MINs)
    data_stats_dict['variances'] = np.array(variances)
    return data_stats_dict


def normalize_data(slicing_windows_wesad: dict) -> dict:
    for i in range(2, bound):
        if i == 12:
            continue
        subject_key = 'S' + str(i)
        subject_dict = dict(slicing_windows_wesad[subject_key])
        for output_key, output_dict in subject_dict.items():
            for data_name, data_stats in dict(output_dict['signals']).items():
                for data, stats in dict(data_stats).items():
                    min_val = np.min(stats)
                    max_val = np.max(stats)
                    data_stats_range = max_val - min_val
                    normalized_data = [(x - min_val) / data_stats_range for x in stats]
                    to_substitute = np.array(normalized_data)
                    slicing_windows_wesad[subject_key][output_key]['signals'][data_name][data] = to_substitute
    return slicing_windows_wesad


if __name__ == '__main__':

    wesad = pd.read_pickle(r'C:\Users\HTTPiego\Documents\WESAD\WESAD\data_set_non_normalizzato.pickle')

    new_wesad = regroup_data(wesad)

    slicing_windows_wesad = get_slicing_windows_wesad(new_wesad)

    normalized_wesad = normalize_data(slicing_windows_wesad)

    model = Sequential()

    '''kernel_regularizer=regularizers.l2(0.01)'''
    # model.add(Dense(512, activation='relu',  input_dim=40))
    # model.add(Dropout(0.5))
    # model.add(Dense(384, activation='relu', input_dim=40))  # , kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu', input_dim=40)) #, kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.3)) #########
    # model.add(Dense(192, activation='relu'))  # , , kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.3))
    # model.add(Dense(128, activation='relu', input_dim=40)) #, , kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.3))
    # model.add(Dense(96, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.3))

    # model.add(Dense(64, activation='relu', input_dim=40))#, kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.2))
    # model.add(Dense(56, activation='relu', input_dim=40))  # , kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.2))
    model.add(Dense(48, activation='relu', input_dim=40))
    # , kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu'))  # , kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation='relu'))  # , kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))  # tolto
    model.add(Dense(8, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2)) #tolto
    model.add(Dense(4, activation='softmax'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    input_data = np.empty(0)
    labels = np.empty(0)
    # for subject in range(2, bound):
    #     if subject == 12 or subject == excluded:
    #         continue
    #     subject_key = 'S' + str(subject)
    #     subject_dict = dict(normalized_wesad[subject_key])
    #     for output_key, output_dict in subject_dict.items():
    #         labels_to_add = np.array(output_dict['labels'])
    #         input_to_add = np.empty(0)
    #         for data_name, data_stats_dict in dict(output_dict['signals']).items():
    #             for stat_name, stat_values in dict(data_stats_dict).items():
    #                 input_to_add = np.concatenate((input_to_add, stat_values), axis=0)
    #
    #         #transpose = np.transpose(input_to_add)
    #
    #         #input_data = np.concatenate((input_data, np.transpose(input_to_add)), axis=0)
    #
    #         input_data = np.concatenate((input_data, input_to_add), axis=0)
    #
    #         labels = np.append(labels, labels_to_add)
    #
    # input_data = input_data.reshape((40, labels.shape[0]))
    # input_data = np.transpose(input_data)
    #
    # #SHUFFLE
    #
    # indices = np.arange(len(labels))
    #
    # random.shuffle(list(indices))
    #
    # labels = np.array([labels[i] for i in indices])
    #
    # input_data = input_data[indices]
    #
    # x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, random_state=42)
    #
    # #split int training and validation
    # x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    #
    # # model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
    #
    # history = model.fit(x_train, y_train,
    #             epochs=100, callbacks=[early_stopping], batch_size=32, validation_data=(x_validation, y_validation))
    #
    # # Extract training accuracy and validation accuracy
    # train_acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # # Plot training accuracy and validation accuracy
    # epochs = range(1, len(train_acc) + 1)
    # plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    #
    # loss, accuracy = model.evaluate(input_data, labels)
    #
    # #predictions = model.predict(input_data)
    #
    # print("input data:\n" + str(np.array(input_data).shape))
    #
    # print("labels:\n" + str(np.array(labels).shape))

    for subject in range(2, bound):
        if subject == 12 or subject == excluded:
            continue
        subject_key = 'S' + str(subject)
        subject_dict = dict(normalized_wesad[subject_key])
        input_data = np.empty(0)
        labels = np.empty(0)
        for output_key, output_dict in subject_dict.items():
            labels_to_add = np.array(output_dict['labels'])
            input_to_add = np.empty(0)
            for data_name, data_stats_dict in dict(output_dict['signals']).items():
                for stat_name, stat_values in dict(data_stats_dict).items():
                    input_to_add = np.concatenate((input_to_add, stat_values), axis=0)

            #input_data = np.concatenate((input_data, np.transpose(input_to_add)), axis=0)

            input_data = np.concatenate((input_data, input_to_add), axis=0)

            labels = np.append(labels, labels_to_add)

        input_data = input_data.reshape((40, labels.shape[0]))
        input_data = np.transpose(input_data)

        indices = np.arange(len(labels))

        random.shuffle(list(indices))

        labels = np.array([labels[i] for i in indices])

        input_data = input_data[indices]

        #split into training and testing
        x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, random_state=42)

        #split int training and validation
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        history = model.fit(x_train, y_train,
                            epochs=100, callbacks=[early_stopping], batch_size=32, validation_data=(x_validation, y_validation))

        # Extract training accuracy and validation accuracy
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        # Plot training accuracy and validation accuracy
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title(subject_key + ' Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        #plt.show()

        loss, accuracy = model.evaluate(input_data, labels)

        #predictions = model.predict(input_data)

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++FINE_TRAINING')

    # FINAL TEST
    excluded_subject_key = 'S' + str(excluded)
    subject_dict = dict(normalized_wesad[excluded_subject_key])
    input_data = np.empty(0)
    labels = np.empty(0)
    for output_key, output_dict in subject_dict.items():
        labels_to_add = np.array(output_dict['labels'])
        input_to_add = np.empty(0)
        for data_name, data_stats_dict in dict(output_dict['signals']).items():
            for stat_name, stat_values in dict(data_stats_dict).items():
                input_to_add = np.concatenate((input_to_add, stat_values), axis=0)

        #input_data = np.concatenate((input_data, np.transpose(input_to_add)), axis=0)

        input_data = np.concatenate((input_data, input_to_add), axis=0)

        labels = np.append(labels, labels_to_add)

    input_data = input_data.reshape((40, labels.shape[0]))

    input_data = np.transpose(input_data)

    # indices = np.arange(len(labels))
    #
    # random.shuffle(list(indices))
    #
    # labels = np.array([labels[i] for i in indices])
    #
    # input_data = input_data[indices]

    print("input data:\n" + str(np.array(input_data).shape))

    print("labels:\n" + str(np.array(labels).shape))

    loss, accuracy = model.evaluate(input_data, labels)

    #predictions = model.predict(input_data)

# 2 76%
# 3 78%
# 4 65%
# 5 85%
# 6 85%
# 7 70%
# 8 72%
# 9 89%
# 10 62%
# 11 91%
# 13 87%
# 14 91%
# 15 80%
# 16 92%
# 17 66%



