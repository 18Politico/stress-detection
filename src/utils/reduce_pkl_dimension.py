import pickle
import os
import re

wesad_path = os.path.abspath('../WESAD')

for d in os.listdir(wesad_path):
    if re.match('S[0-9]', d):
        current_S = os.path.join(wesad_path, d)
        print(current_S)
        current_pkl = os.path.join(current_S, d + '.pkl')
        tmp_pkl = os.path.join(current_S, 'tmp.pkl')
        # reducing pkl dimension
        with open(os.path.join(current_pkl), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            with open(tmp_pkl, 'wb') as custom:
                pickle.dump(data, custom, pickle.HIGHEST_PROTOCOL)
        # remove the old pkl and rename the new pkl
        os.remove(current_pkl)
        os.rename(tmp_pkl, current_pkl)
