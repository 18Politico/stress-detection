import os

ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.path.abspath(__file__)))))
WESAD_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'dataset', 'WESAD')
RAW_DATA_PKL = os.path.join(ROOT_DIRECTORY, 'dataset', 'raw_data.pkl')
