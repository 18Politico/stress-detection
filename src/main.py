from data_preprocessing.data_manager import *
import numpy as np

manager = DataManager()
# make the training reproducible
manager.set_random_seed()
# prepare data for training
print('loading raw data...')
data = manager.load_raw_data()
print('formatting raw data...')
data = manager.reformat_raw_data(data)
print('preprocessing raw data...')
data = manager.preprocess_raw_data(data)
accuracy_scores = []
input_shape = (161, 15, 1)

# train cnn
for i, data_validation in enumerate(data):
    data_training = np.concatenate((data[:i], data[i + 1:]))
    x, y = manager.prepare_data_for_cnn(data_training)
    x_validation, y_validation = manager.prepare_data_for_cnn([data_validation])
    loss, accuracy = manager.train_cnn(x, y, x_validation, y_validation, input_shape)
    accuracy_scores.append(accuracy)

# Calculate the average performance
print('accuracy mean: ', np.mean(accuracy_scores))
print('accuracy std', np.std(accuracy_scores))
