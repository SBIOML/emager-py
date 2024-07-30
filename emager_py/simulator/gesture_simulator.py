from emager_py.simulator.data_simulator import EmagerSimulator
from emager_py.simulator.virtual_port import *
import re

class GestureSimulator:
    def __init__(self, datasetpath, sampling, PORT2, BAUDRATE):

        self.path = datasetpath
        self.simulator = None
        self.odh = OfflineDataHandler()
        self.max_reps =  self.find_max_reps()
        self.prepare_data()
        self.simulator = EmagerSimulator(prepared_data, sampling, PORT2, BAUDRATE)
        
    def find_max_reps(self):
        # Initialize a list to hold the repetition numbers
        reps = []
        # Loop through the files in the folder
        for filename in os.listdir(self.path):
            if filename.endswith("_emg.csv"):
                # Extract the repetition number using regular expression
                match = re.search(r"_R_(\d+)_emg\.csv", filename)
                if match:
                    rep_number = int(match.group(1))
                    reps.append(rep_number)
        return max(reps)

    def prepare_data(self, classes=None, num_reps=None):
        if num_reps is None:
            num_reps = self.max_reps
        classes_values = classes
        classes_regex = make_regex(left_bound = "C_", right_bound="_", values = classes_values)
        reps_values = [str(num) for num in range(num_reps)]
        reps_regex = make_regex(left_bound = "R_", right_bound="_emg.csv", values = reps_values)
        dic = {
            "classes": classes_values,
            "classes_regex": classes_regex,
            "reps": reps_values,
            "reps_regex": reps_regex,
        }
        self.odh.get_data(folder_location=self.path, filename_dic=dic, delimiter=",")
        if self.simulator:
            self.simulator.prepare_data(np.array(self.odh.data))
        return np.array(self.odh.data)

    def simulate(self):
        return self.gesture