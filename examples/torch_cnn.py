import numpy as np
import torch
import lightning as L
from sklearn.metrics import accuracy_score

import emager_py.torch.datasets as etd
import emager_py.utils.utils as eutils
import emager_py.data.transforms as etrans
import emager_py.data.dataset as ed
import emager_py.data.data_processing as edp
import emager_py.torch.models as etm

eutils.set_logging()

DATASETS_PATH = "/Users/gabrielgagne/Documents/Datasets/EMAGER/"
TRAIN_SUBJECT = 0
CROSS_SUBJECT = 1

"""
emager-py relies builds on Lightning AI for its PyTorch integration, 
educing massively the amount of boilerplate code needed to train models.
Refer to their documentation for more usage information.
"""

# First, create the torch DataLoaders
train, test = etd.get_lnocv_dataloaders(
    DATASETS_PATH,
    TRAIN_SUBJECT,
    "001",
    9,
    transform=etrans.default_processing,
)

# Now, instantiate a mnodel (or load it from disk if you wish)
model = etm.EmagerCNN((4, 16), 6, -1)

# Finally, Lightning takes care of the rest!
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, train)
trainer.test(model, test)

print("*" * 80)
print("Now, the PyTorch model is trained and ready to be used in your experiment.")
print("In this case, let's try to evaluate it on a different subject.")
print("*" * 80)

# Let's test on another subject
data = ed.load_emager_data(DATASETS_PATH, CROSS_SUBJECT, 1)
processed = etrans.default_processing(data)
data, labels = edp.extract_labels(processed)

data = torch.from_numpy(data).float()
preds = np.argmax(model(data).cpu().detach().numpy(), axis=1)
xacc = accuracy_score(labels, preds)

print(f"Cross-subject accuracy: {xacc*100:.2f}%")
