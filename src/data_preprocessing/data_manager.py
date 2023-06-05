import os
from utils.paths import *

print(path)
class DataManager:
    SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    WESAD_DIRECTORY = os.path.join(os.path.abspath(__file__), '..', 'dataset', 'WESAD')

    def load_data(self):
        for id in self.SUBJECTS:
            self.__load_subject_data(id)

    def __load_subject_data(self, id):
        pass
