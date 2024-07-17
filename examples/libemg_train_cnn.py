
from libemg.data_handler import OfflineDataHandler
from libemg.utils import make_regex
from libemg.feature_extractor import FeatureExtractor
from libemg.filtering import Filter

import torch
from torch.utils.data import DataLoader, TensorDataset
import emager_py.torch.models as etm
import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = "C:\GIT\Datasets/Libemg/"
SESSION = "Demo"
DATASETS_PATH = f"{BASE_PATH}{SESSION}/"
SAVE_PATH = f"{BASE_PATH}{SESSION}/"

NUM_CLASSES = 5
NUM_REPS = 5
WINDOW_SIZE = 200
WINDOW_INCREMENT = 10
EPOCH = 10

def prepare_data(dataset_folder):
        classes_values = [str(num) for num in range(NUM_CLASSES)]
        classes_regex = make_regex(left_bound = "C_", right_bound="_", values = classes_values)
        reps_values = [str(num) for num in range(NUM_REPS)]
        reps_regex = make_regex(left_bound = "R_", right_bound="_emg.csv", values = reps_values)
        dic = {
            "classes": classes_values,
            "classes_regex": classes_regex,
            "reps": reps_values,
            "reps_regex": reps_regex,
        }
        odh = OfflineDataHandler()
        odh.get_data(folder_location=dataset_folder, filename_dic=dic, delimiter=",")
        filter = Filter(1000)
        notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
        filter.install_filters(notch_filter_dictionary)
        bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
        filter.install_filters(bandpass_filter_dictionary)
        filter.filter(odh)
        return odh

data = prepare_data(DATASETS_PATH)
# for i in range(len(data.data)):
#     plt.plot(data.data[i])
#     plt.show()

# Split data into training and testing
train_data = data.isolate_data("reps", [0,1,2])
test_data = data.isolate_data("reps", [3,4])

# Extract windows 
train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)

print(f"Training metadata: {train_meta}, Testing metadata: {test_meta}")
print(f"Training windows: {train_windows.shape}, Testing windows: {test_windows.shape}")


# Features extraction
# Extract MAV since it's a commonly used pipeline for EMG
fe = FeatureExtractor()
train_data = fe.getMAVfeat(train_windows)
train_labels = train_meta["classes"]
test_data = fe.getMAVfeat(test_windows)
test_labels = test_meta["classes"]

# pause for visualize features
features_data = {"key": train_data}
fe.visualize_feature_space(features_data, "PCA", classes=train_labels)

train_dl = DataLoader(
    TensorDataset(torch.from_numpy(train_data.astype(np.float32)), torch.from_numpy(train_labels)),
    batch_size=64,
    shuffle=True,
)
test_dl = DataLoader(
    TensorDataset(torch.from_numpy(test_data.astype(np.float32)), torch.from_numpy(test_labels)),
    batch_size=256,
    shuffle=False,
)

# Fit and test the model
classifier = etm.EmagerCNN((4, 16), NUM_CLASSES, -1)

res = classifier.fit(train_dl, test_dl, max_epochs=EPOCH)
acc = int(res[0]["test_acc"]*1000)
print(f"Resultat: {res} accuracy : {acc}/1000")

# Save the model
model_path = f"{SAVE_PATH}libemg_torch_cnn_{SESSION}_{acc}.pth"
torch.save(classifier.state_dict(), model_path)
print(f"Model saved at {model_path}")