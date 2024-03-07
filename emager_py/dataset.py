import numpy as np
import emager_py.data_processing as dp
import emager_py.quantization as dc
import emager_py.utils as eutils
import logging as log
import os


def load_emager_data(path, subj, ses, differential=False):
    """
    Load EMG data from EMaGer v1 dataset.

    Params:
        - path : path to EMaGer root
        - user_id : subject id
        - session_nb : session number

    Returns the raw loaded data (0-65535) with shape (nb_gesture, nb_repetition, samples, num_channels)
    """

    # Parameters
    # user_id = "001"
    # session_nb = "000"
    nb_gesture = 6
    nb_repetition = 10
    nb_pts = 5000
    start_path = "%s/%s/session_%s/" % (path, subj, ses)
    data_array = np.zeros((nb_gesture, nb_repetition, 64, nb_pts), dtype=int)

    first_file = os.listdir(start_path)[0]
    arm_used = "right" if "right" in first_file else "left"
    for gest in range(nb_gesture):
        for rep in range(nb_repetition):
            path = (
                start_path
                + subj
                + "-"
                + ses
                + "-00"
                + str(gest)
                + "-00"
                + str(rep)
                + "-"
                + arm_used
                + ".csv"
            )
            one_file = np.transpose(np.loadtxt(path, delimiter=","))
            data_array[gest, rep, :, :] = one_file[:, -nb_pts:]
    if differential:
        data_array = np.reshape(data_array, (nb_gesture, nb_repetition, 16, 4, nb_pts))
        final_array = data_array[:, :, :, 0:3, :] - data_array[:, :, :, 1:4, :]
        final_array = np.reshape(final_array, (nb_gesture, nb_repetition, 48, nb_pts))
    else:
        final_array = data_array
    final_array = np.swapaxes(final_array, 2, 3)
    log.info(f"Loaded subject {subj} session {ses}. Data shape {final_array.shape}.")
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


def get_processed_data_array(path, subj, ses) -> np.ndarray:
    """
    Load and process a data array from parameters.

    Returns an array of shape (n_samples, 65), where the last column are the labels
      and the rest are the processed EMG channels' signals
    """
    data_array = load_emager_data(path, subj, ses)
    processed = dp.preprocess_data(data_array)
    compressed = dc.nroot_c(processed, 2.5, 20000)
    X, y = dp.extract_with_labels(compressed)

    X_rolled = dp.roll_data(X, 2)

    # Copy the labels to be the same size as the data
    nb_rolled = int(np.floor(X_rolled.shape[0] / y.shape[0]))
    y_rolled = np.tile(y, nb_rolled)
    y_rolled = np.array(y_rolled, dtype=np.uint8)

    log.info(
        f"Loaded processed subject {subj} session {ses}. Data shape {X_rolled.shape}, labels shape {y_rolled.shape}."
    )

    return np.hstack((X_rolled, y_rolled.reshape((-1, 1))))


def get_train_test_data_arrays(path, subj, ses, split=0.8) -> np.ndarray:
    """
    Get train and test datasets from parameters. Loads the data from disk.

    Returns a tuple: `(train_ds`, `test_ds)`. Both datasets are of shape (n_samples, 65),
    where the last column are the labels and the rest are the processed EMG channels' signals
    """
    ds = get_processed_data_array(path, subj, ses)

    np.random.shuffle(ds)  # in-place

    ds_len = len(ds)
    ds_split = int(ds_len * split)

    train_ds = ds[0:ds_split]
    test_ds = ds[ds_split:]

    log.info(f"Train preprocessed dataset len : {len(train_ds)}")
    log.info(f"Test preprocessed dataset len : {len(test_ds)}")

    return train_ds, test_ds


def load_processed_data(path, subj, ses, train=False) -> np.ndarray:
    log.info(f"Attempting to load subject {subj} session {ses} from npz")
    t = "train" if train else "test"
    return np.load(f"{path}/{subj}_{ses}_{t}.npz")


def process_then_save_all_test_train(dataset_path, out_path, transform=None):
    subjects = get_subjects()
    sessions = get_sessions()

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for subject in subjects:
        for session in sessions:
            train, test = get_train_test_data_arrays(
                dataset_path,
                subject,
                session,
            )

            train_data = train[:, :-1]
            train_label = train[:, -1]

            test_data = test[:, :-1]
            test_label = test[:, -1]

            if transform is not None:
                train_data, test_data = transform(train_data), transform(test_data)

            trn_name = f"{out_path}/{subject}_{session}_train.npz"
            test_name = f"{out_path}/{subject}_{session}_test.npz"

            log.info(f"Saving subject {subject} session {session} to {trn_name}")
            np.savez(
                trn_name,
                emg=train_data,
                label=train_label,
            )
            log.info(f"Saving subject {subject} session {session} to {test_name}")
            np.savez(
                test_name,
                emg=test_data,
                label=test_label,
            )


def get_subjects(path):
    """
    List all subjects in EMaGer dataset.

    TODO: Add rotation experiments
    """

    def filt(d):
        try:
            int(d)
            return True
        except ValueError:
            return False

    return list(filter(lambda d: filt(d), os.listdir(path)))


def get_sessions():
    """
    Get the sessions from EMaGer dataset.

    TODO: Add rotation experiments
    """
    return ["001", "002"]


def get_repetitions():
    """
    Get the number of repetitions for every gesture.
    """
    return 10
