
import emager_py.data.data_processing as dp
import emager_py.data.dataset as ed
import emager_py.data.transforms as etrans
import numpy as np


DATASET_PATH = "C:\GIT\Datasets\EMAGER/"
data_array = ed.load_emager_data(
    DATASET_PATH, "000", "002", differential=False
)

data = dp.prepare_shuffled_datasets(
    data_array, split=0.8, absda="train", transform=etrans.default_processing
)

emb = np.random.rand(10, 64)
class_emb = np.random.rand(6, 64)
closest_class = dp.cosine_similarity(emb, class_emb, closest_class=True)
print(closest_class, closest_class.shape)
class_similarity = dp.cosine_similarity(emb, class_emb, closest_class=False)
print(class_similarity, class_similarity.shape)

train_emg, test_emg = ed.get_lnocv_datasets(
    DATASET_PATH, 0, 1, 9
)
print(train_emg.shape, test_emg.shape)
emg, labels = dp.extract_labels(test_emg)
print(emg.shape, labels.shape)
anchor, pos, neg = dp.generate_triplets(emg, labels, 1000)
print(anchor.shape, pos.shape, neg.shape)