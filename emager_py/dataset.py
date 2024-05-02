"""
EMaGer dataset routines.

This dataset contains data from 13 subjects, labeled 0 to 12.

The dataset is structured in the following way:

`<EMAGER>/<subject_id>/<session>/<filename>.csv`

Where:

- `<subject_id>` is the subject ID from 000 to 012
- `<session>` is the session number, usually either `session_0` or `session_1`. There are some special session which are not yet supported.
- `<filename>.csv`, ie `002-001-003-009-right.csv`:
  - `002` is the subject ID
  - `001` is the session number
  - `003` is the gesture id
  - `009` is the segment number for the given gesture
  - `right` recording done on the right forearm

NOTE: subject, gesture and repetition numbers are 0-indexed. Session is 1-indexed.

This module provides routines to load, process and save EMaGer dataset-compatible data. 

The saving/loading routines expect the data arrays to have the following shape:
- (nb_gesture, nb_repetition, samples, num_channels), denoted as (G, R, N, C) OR (G, R, N, H, W) for images, where C = H*W

Usually, the entry point is `load_emager_data`, which can load any subject and session from the dataset including pre-processed data.
Then, use functions from `emager_py.data_processing` to process the data, extract the labels, shuffle in batches the dataset, etc.
"""

import numpy as np
import logging as log
import os

import emager_py.data_processing as dp
import emager_py.utils as eutils
import emager_py.transforms as etrans

_DATASET_TEMPLATE = {
    "subject": "%s/",
    "session": "session_%s/",
    "repetition": "%s-%s-%s-%s-%s.csv",
}


def format_subject(subject: int | str) -> str:
    if isinstance(subject, str):
        subject = int(subject)
    return _DATASET_TEMPLATE["subject"] % str(subject).zfill(3)


def format_session(session: int | str) -> str:
    if isinstance(session, str):
        session = int(session)
    return _DATASET_TEMPLATE["session"] % str(session).zfill(3)


def format_repetition(subject, session, gesture, repetition, arm="right") -> str:
    if isinstance(subject, str):
        subject = int(subject)
    if isinstance(session, str):
        session = int(session)
    if isinstance(gesture, str):
        gesture = int(gesture)
    if isinstance(repetition, str):
        repetition = int(repetition)
    if len(arm) == 0:
        template = "".join(_DATASET_TEMPLATE["repetition"].rsplit("-", 1))
    else:
        template = _DATASET_TEMPLATE["repetition"]

    return template % (
        str(subject).zfill(3),
        str(session).zfill(3),
        str(gesture).zfill(3),
        str(repetition).zfill(3),
        arm,
    )


def get_subjects(path) -> list[str]:
    """
    List all subjects in EMaGer dataset.
    """

    def filt(d):
        try:
            int(d)
            return True
        except ValueError:
            return False

    subjects = list(filter(lambda d: filt(d), os.listdir(path)))
    subjects.sort()
    return subjects


def get_sessions() -> list[str]:
    """
    Get the sessions from EMaGer dataset.

    TODO: Add rotation experiments
    """
    return ["001", "002"]


def get_repetitions() -> list[str]:
    """
    Get the available repetitions for every gesture.

    TODO: actually read from disk what repetitions exist.
    """
    return [f"{i:03d}" for i in range(10)]


def load_emager_data(dataset_path, subject, session, differential=False, floor_to=100):
    """
    Load EMG data from EMaGer v1 dataset.

    Params:
        - path : path to EMaGer dataset root, or any emager-complying dataset.
        - user_id : subject id
        - session_nb : session number

    Returns the raw u16 data formatted as GRNC.
    """

    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} not found."

    base_path = dataset_path + format_subject(subject) + format_session(session)
    log.info(base_path)

    files = [f.split("-") for f in os.listdir(base_path)]
    gesture_toks = [int(f[2]) for f in files]
    rep_toks = [int(f[3].split(".")[0]) for f in files]

    nb_gesture = max(gesture_toks) + 1
    nb_repetition = max(rep_toks) + 1
    nb_pts = 10000000

    first_file = os.listdir(base_path)[0]
    arm_used = (
        "right" if "right" in first_file else "left" if "left" in first_file else ""
    )
    gest_rep_arrays = []
    for gest in range(nb_gesture):
        for rep in range(nb_repetition):
            dataset_path = base_path + format_repetition(
                subject, session, gest, rep, arm=arm_used
            )
            new_data = np.loadtxt(dataset_path, delimiter=",")
            gest_rep_arrays.append(new_data)
            if len(new_data) < nb_pts:
                nb_pts = len(new_data)
    nb_pts = nb_pts - (nb_pts % floor_to)
    data_array = np.zeros((nb_gesture, nb_repetition, 64, nb_pts), dtype=int)

    for gest in range(nb_gesture):
        for rep in range(nb_repetition):
            one_file = np.transpose(gest_rep_arrays[gest * nb_repetition + rep])
            data_array[gest, rep, :, :] = one_file[:, -nb_pts:]

    if differential:
        data_array = np.reshape(data_array, (nb_gesture, nb_repetition, 16, 4, nb_pts))
        final_array = data_array[:, :, :, 0:3, :] - data_array[:, :, :, 1:4, :]
        final_array = np.reshape(final_array, (nb_gesture, nb_repetition, 48, nb_pts))
    else:
        final_array = data_array
    final_array = np.swapaxes(final_array, 2, 3)
    log.info(
        f"Loaded subject {subject} session {session}. Data shape {final_array.shape}."
    )
    return final_array


def generate_processed_validation_data(
    emager_path: str,
    subject: str,
    session: str,
    transform_fn,
    output_dir: str = "./",
    save=True,
):
    """
    Generate some processed data for dataflow and FINN validation.

    Saves numpy arrays from `subject` `session` under `output_dir`:
        - `finn_preproc_valid_data.npz` : entire processed validation `data` and `labels`

    Returns a tuple of generated data: (data, labels), and saves the numpy arrays to disk in `output_dir`
    """

    # Generate processed data
    data = load_emager_data(emager_path, subject, session)
    data = transform_fn(data)
    data, labels = dp.extract_labels(data)
    data, labels = dp.shuffle_dataset(data, labels, 1)
    data = data.astype(np.float32)

    if save:
        np.savez(output_dir + "/finn_preproc_valid_data", data=data, labels=labels)

    return data, labels


def generate_raw_validation_data(
    emager_path: str,
    subject: str,
    session: str,
    transform_fn,
    output_dir: str = "./",
    save: bool = True,
    n_samples: int = 25,
):
    """
    Generate some raw validation data. Load the subject, extracts labels and shuffles the dataset.

    Saves arrays from `subject` `session` under `output_dir`:
        - `finn_raw_valid_data.npz` : raw validation `data` and `labels`, shapes (n_samples, 64) and (n_samples, 1)
        - `finn_transform_fn_name.txt` : transform fn name for validating raw data

    Returns a tuple of generated data: (data, labels), and saves the numpy arrays to disk in `output_dir`
    """

    decim = eutils.get_transform_decimation(transform_fn)

    # Save raw validation data
    data = load_emager_data(emager_path, subject, session)
    raw_data, labels = dp.extract_labels(data)
    raw_data, labels = dp.shuffle_dataset(raw_data, labels, decim)
    raw_data = raw_data[: n_samples * decim]
    labels = labels[: n_samples * decim : decim]

    if save:
        np.savez(
            output_dir + "/finn_raw_valid_data",
            data=raw_data,
            labels=labels,
        )

        with open(output_dir + "/finn_transform_fn_name.txt", "w") as f:
            f.write(transform_fn.__name__)

    return raw_data, labels


def process_save_dataset(
    data, out_path: str, transform: callable, subject, session
) -> str:
    """
    Process and save a dataset as numpy array to disk.

    Parameters:
        - `data`: np array of shape (n_gestures, n_reps, n_samples, n_channels)
        - `out_path`: path to save the processed dataset

    The `transform` must take in a Numpy array of shape (n_samples, n_channels) and return the transformed array with the same dimensions.

    The processed dataset has the exact same folder structure as the original EMaGer dataset.
    """

    if len(data.shape) == 2:
        data = np.reshape(data, (1, 1, *data.shape))
    elif len(data.shape) != 4:
        raise ValueError(
            "Data shape must be (n_gestures, n_reps, n_samples, n_channels)"
        )

    out_path = out_path + format_subject(subject) + format_session(session)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    nb_gesture = data.shape[0]
    nb_rep = data.shape[1]

    for gesture in range(nb_gesture):
        for rep in range(nb_rep):
            processed_data = transform(data[gesture, rep, :, :])
            np.savetxt(
                out_path + format_repetition(subject, session, gesture, rep, ""),
                processed_data,
                delimiter=",",
            )

    log.info(f"Saved processed dataset to {out_path}")
    return out_path


def load_process_save_dataset(
    dataset_path: str, out_path: str, transform: callable, subjects=None, sessions=None
):
    """
    Load EMaGer-compatible dataset from disk, process it and save it back to disk with the same folder structure.

    The `transform` must take in a Numpy array of shape (n_samples, n_channels) and return the transformed array with the same dimensions.

    Returns the new dataset root path.
    """

    if isinstance(transform, str):
        transform = etrans.transforms_lut[transform]

    if subjects is None:
        subjects = get_subjects(dataset_path)
    elif not isinstance(subjects, list):
        subjects = [subjects]

    if not sessions:
        sessions = get_sessions()
    elif isinstance(sessions, int):
        sessions = [sessions]

    for subject in subjects:
        for session in sessions:
            session_data = load_emager_data(dataset_path, subject, session)
            ret = process_save_dataset(
                session_data, out_path, transform, subject, session
            )
            log.info(
                f"Saved processed dataset for subject {subject} session {session} at {ret}"
            )
    return out_path


def get_intersession_cv_datasets(dataset_path: str, subject):
    """
    Get an inter-session cross-validation dataset.

    This can easily be used as a train/test dataset.

    Returns a tuple of two datasets: (train, test)
    """

    sessions = get_sessions()
    return tuple([load_emager_data(dataset_path, subject, s) for s in sessions])


def get_lnocv_datasets(
    dataset_path: str,
    subject: str | int,
    session: int | list[int],
    test_rep: int | list[int],
):
    """
    Get the Leave-N-Out Cross Validation (LNOCV) datasets from EMaGer dataset.

    This can easily be used as a train/test dataset.

    Params:
        - `dataset_path`: path to EMaGer dataset root
        - `subject`: subject ID
        - `session`: session number(s).
        - `test_rep`: repetition(s) to leave out as testing/validation set

    Returns a tuple of datasets: (train, test), each as GRNC arrays.
    """

    if not isinstance(test_rep, list):
        test_rep = [test_rep]

    # First, load the entire dataset
    if not isinstance(session, list):
        session = [session]

    # concatenate them as repetitions
    data = [load_emager_data(dataset_path, subject, ses) for ses in session]
    test_rep = [
        int(r) + k * data[0].shape[1] for k in range(len(data)) for r in test_rep
    ]
    data = np.concatenate(data, axis=1)

    # extract the LOOCV datasets
    lo_data = data[:, test_rep, :, :]

    # delete extracted data from the original dataset
    data = np.delete(data, test_rep, axis=1)

    log.info(
        f"Loaded LOOCV datasets for subject {subject} session {session} with shapes {data.shape}, {lo_data.shape}."
    )
    return data, lo_data


if __name__ == "__main__":
    from emager_py import utils
    from emager_py import transforms

    utils.DATASETS_ROOT = "/home/gabrielgagne/Documents/Datasets/EMAGER/"

    utils.set_logging()
    print(get_subjects(utils.DATASETS_ROOT))
    """
    processed_path = load_process_save_dataset(
        "/home/gabrielgagne/Documents/Datasets/EMAGER/",
        "/home/gabrielgagne/Documents/Datasets/PEMAGER",
        transforms.root_processing,
        [0, 1, 2],
        1,
    )
    """
    processed_path = "/home/gabrielgagne/Documents/Datasets/PEMAGER/root_processing/"
    load_emager_data(processed_path, 1, 1)
    """
    print(
        get_subjects("/home/gabrielgagne/Documents/Datasets/EMAGER/"),
        get_sessions(),
        get_repetitions(),
    )
    """
    d, lo = get_lnocv_datasets(utils.DATASETS_ROOT, 3, 1, 4)
    d, lo = get_lnocv_datasets(utils.DATASETS_ROOT, 3, 1, [4, 5])
    d, lo = get_lnocv_datasets(utils.DATASETS_ROOT, 3, [1, 2], [4, 5])
    print(d.shape, lo.shape)
    """
    d, lo = get_intersession_cv_datasets(
        utils.DATASETS_ROOT, 0
    )
    print(d.shape, lo.shape)
    """
