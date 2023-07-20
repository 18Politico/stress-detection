from data_preprocessing.data_manager import *
import numpy as np

with open(PREPROCESSED_DATA, 'rb') as f:
    manager = DataManager()
    # make the training reproducible
    manager.set_random_seed()
    # load data already preprocessed
    data = pickle.load(f)
    accuracy_scores = []
    input_shape = (161, 15, 1)
    for i, data_validation in enumerate(data):
        data_training = np.concatenate((data[:i], data[i + 1:]))
        x, y = manager.prepare_data_for_cnn(data_training)
        x_validation, y_validation = manager.prepare_data_for_cnn([data_validation])
        loss, accuracy = manager.train_cnn(x, y, x_validation, y_validation, input_shape)
        accuracy_scores.append(accuracy)

    # Calculate the average performance across all folds
    print('accuracy mean: ', np.mean(accuracy_scores))
    print('accuracy std', np.std(accuracy_scores))
