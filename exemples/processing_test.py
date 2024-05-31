from emager_py.data_processings import data_processing as dp
from emager_py.data_processings import dataset as ed
from emager_py.utils import utils
import numpy as np


# generate sinus of 60 hz
ret = dp.filter_utility(np.zeros((1000, 10)))

data_path = "C:\GIT\Datasets\EMAGER"
data_array = ed.load_emager_data(
    data_path, "000", "002", differential=False
)
print(data_array.shape)
# preprocess the data
averages = dp.preprocess_data(data_array)
print(np.shape(averages))

# Visualize the data

# roll the data
rolled = dp.roll_data(averages, 2)
print(np.shape(rolled))
X, y = dp.extract_labels(data_array)

y = np.array(y, dtype=np.uint8)
print(np.shape(X))
_TIME_LENGTH = 25
_VOTE_LENGTH = 150
nb_votes = int(np.floor(_VOTE_LENGTH / _TIME_LENGTH))

expected = np.array(
    [
        np.argmax(np.bincount(y[i : i + _TIME_LENGTH]))
        for i in range(0, len(y), _TIME_LENGTH)
    ]
)
maj_expected = np.array(
    [
        np.argmax(np.bincount(expected[i : i + nb_votes]))
        for i in range(0, len(expected), nb_votes)
    ]
)

print(np.shape(expected))
print(np.shape(maj_expected))

"""
example usage (sd.save_training_data)
data_array = getData_EMG(dataset_path, subject, session, differential=False)
averages_data = dp.preprocess_data(data_array)
compressed_data = dp.compress_data(averages_data, method=compressed_method)
rolled_data = dp.roll_data(compressed_data, 2)
X, y = dp.extract_with_labels(rolled_data)
"""

# TODO : Finalize 
