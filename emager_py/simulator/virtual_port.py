from emager_py.simulator.data_simulator import EmagerSimulator
from libemg.data_handler import OfflineDataHandler
from libemg.utils import make_regex
import subprocess as sp
import os
import numpy as np
import sys
import time



def virtual_port(prepared_data:np.array, sampling):
    PORT1 = '/dev/ttyV1' if sys.platform.startswith('linux') else 'COM1'
    PORT2 = '/dev/ttyV2' if sys.platform.startswith('linux') else 'COM2'
    BAUDRATE = 1500000

    if sys.platform.startswith('linux'):
        proc = sp.Popen(
            [
                "socat",
                f"pty,rawer,link={PORT1}",
                f"pty,rawer,link={PORT2}",
            ]
        )
        time.sleep(3)

    # streamdata
    simulator = EmagerSimulator(prepared_data, sampling, PORT2, BAUDRATE)
    simulator.start()

    return PORT1, simulator

def virtual_port_libemg_v1(sampling, datasetpath, num_classes, num_reps) -> str:
    '''
    Uses files with the following naming convention:
    R_{rep}_{class}.csv
    '''
    classes_values = [str(num) for num in range(num_classes)]
    classes_regex = make_regex(left_bound = "R_", right_bound=".csv", values = classes_values)
    reps_values = [str(num) for num in range(num_reps)]
    reps_regex = make_regex(left_bound = "C_", right_bound="_", values = reps_values)
    dic = {
        "classes": classes_values,
        "classes_regex": classes_regex,
        "reps": reps_values,
        "reps_regex": reps_regex,
    }
    odh = OfflineDataHandler()
    odh.get_data(folder_location=datasetpath, filename_dic=dic, delimiter=",")
    data = np.array(odh.data)
    return virtual_port(data, sampling)

def virtual_port_libemg_v2(sampling, datasetpath, num_classes, num_reps) -> str:
    '''
    Uses files with the following naming convention:
    C_{class}_R_{rep}_emg.csv
    '''
    classes_values = [str(num) for num in range(num_classes)]
    classes_regex = make_regex(left_bound = "C_", right_bound="_", values = classes_values)
    reps_values = [str(num) for num in range(num_reps)]
    reps_regex = make_regex(left_bound = "R_", right_bound="_emg.csv", values = reps_values)
    dic = {
        "classes": classes_values,
        "classes_regex": classes_regex,
        "reps": reps_values,
        "reps_regex": reps_regex,
    }
    odh = OfflineDataHandler()
    odh.get_data(folder_location=datasetpath, filename_dic=dic, delimiter=",")
    return virtual_port(odh.data, sampling)


def virtual_port_emager(sampling, datasetpath, num_classes, num_reps, arm="right") -> str:
    '''
    Uses files with the following naming convention:
    {user}-{session}-{classes}-{reps}.csv
    '''
    # Normalize path to use '/' as the separator
    normalized_path = os.path.normpath(datasetpath)
    folders = normalized_path.split(os.path.sep)
    print(folders)
    session_id = folders[-1].split("_")[-1]
    subject_id = folders[-2]
    print(f"Subject: {subject_id}, Session: {session_id}")
    left_bound = f"{subject_id}-{session_id}-00"
    classes_values = [str(num) for num in range(num_classes)]
    classes_regex = make_regex(left_bound = left_bound, right_bound="", values = classes_values)
    reps_values = [str(num) for num in range(num_reps)]
    reps_regex = make_regex(left_bound = "", right_bound=f"-{arm}.csv", values = reps_values)
    dic = {
        "classes": classes_values,
        "classes_regex": classes_regex,
        "reps": reps_values,
        "reps_regex": reps_regex,
    }
    odh = OfflineDataHandler()
    odh.get_data(folder_location=datasetpath, filename_dic=dic, delimiter=",")
    return virtual_port(odh.data, sampling)
