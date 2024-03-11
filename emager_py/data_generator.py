import numpy as np
import time
import threading
import logging as log

import emager_py.utils as utils
import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.emager_redis as er


class EmagerDataGenerator:
    def __init__(self, host: str, dataset_root: str, shuffle: bool = True):
        """
        This class allows you to simulate a live sampling process. It does not apply any SigProc.
        It is meant to run on a host PC, not on an embedded device.

        To sync the parameters such as sample generation rate and batch size, call `update_params()`.

        First, `prepare_data` to load some data.

        Then, call `DataGenerator.serve_data(...)` to start serving data,
        or `get_serve_thread(...)` to get a server thread which can then be `start()`ed.

        Parameters:
            - host: Redis hostname to connect to
            - dataset_root: EMaGer dataset root
            - shuffle: shuffle the data before serving
        """
        self.__dataset_root = dataset_root
        self.__r = er.EmagerRedis(host)

        self.batch = 1
        self.sampling_rate = 1000

        self.emg = np.ndarray((0, 64))
        self.labels = np.ndarray((1))

        self.shuffle = shuffle

    def prepare_data(self, subject: str, session: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data from disk and prepare it for serving, including shuffling if `self.shuffle` is True.

        Sets Redis `emager_utils.GENERATED_SAMPLES_KEY` key.
        """
        assert (
            session in ed.get_sessions()
        ), f"Requested session is invalid ({session})."

        dat = ed.load_emager_data(self.__dataset_root, subject, session).astype(
            np.int16
        )
        emg, labels = dp.extract_labels(dat)
        if self.shuffle:
            emg, labels = dp.shuffle_dataset(
                emg, labels, int(self.__r.get(self.__r.BATCH_KEY))
            )

        self.emg = emg.astype(np.int16)
        self.labels = labels.astype(np.uint8)

        self.__r.set(self.__r.GENERATED_SAMPLES_KEY, len(self))

        log.info(
            f"Prepared data of shape {self.emg.shape}, labels {self.labels.shape}."
        )
        return self.emg, self.labels

    def generate_data(
        self,
    ):
        """
        Create a data generator.
        """
        for i in range(len(self) // self.batch):
            emg = self.emg[self.batch * i : self.batch * (i + 1), :]
            labels = self.labels[self.batch * i : self.batch * (i + 1)]
            yield emg, labels

    def serve_data(self):
        """
        For loop over `self.generate_data` which pushes 1 data batch to `self.__redis` in a timely manner.
        For a threaded usage, prefer `self.get_serve_thread` instead.
        """
        self.update_params()
        lpush_time = self.batch / self.sampling_rate
        log.info(f"Serving data every {lpush_time:.4f} s")
        # rep_start_time = time.perf_counter()
        for emg, label in self.generate_data():
            t0 = time.perf_counter()
            p = self.__r.r.pipeline()
            p.lpush(self.__r.SAMPLES_FIFO_KEY, emg.tobytes())
            p.lpush(self.__r.LABELS_FIFO_KEY, label.tobytes())
            p.execute()
            dt = time.perf_counter() - t0
            if dt < lpush_time:
                time.sleep(lpush_time - dt)
            # print(time.perf_counter() - t0)

    def clear_data(self):
        self.__r.clear_data()

    def get_serve_thread(self):
        """
        Return a Thread which is ready to start serving data.

        Call `t.start()` on this function's return value to start serving.
        """
        return threading.Thread(target=self.serve_data)

    def update_params(self):
        """
        Update data generation parameters from Redis.
        """
        self.batch = int(self.__r.get(self.__r.BATCH_KEY))
        self.sampling_rate = int(self.__r.get(self.__r.FS_KEY))

        log.info(
            f"Parameters updated from Redis: batch size {self.batch}, fs {self.sampling_rate}"
        )

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    utils.set_logging()
    host = er.get_docker_redis_ip()
    batch = 75

    r = er.EmagerRedis(host)
    r.flushall()
    r.set_sampling_params(1000, batch)

    dg = EmagerDataGenerator(host, utils.DATASETS_ROOT + "EMAGER/", True)
    emg, lab = dg.prepare_data("004", "001")

    dg.get_serve_thread().start()
    print("Len of generated data: ", r.get(r.GENERATED_SAMPLES_KEY))

    for i in range(len(lab)):
        data, labels = r.brpop_sample()
        batch = len(labels)
        print(f"Received shape {data.shape}")
        assert np.array_equal(data, emg[batch * i : batch * (i + 1)]), print(
            data, emg[batch * i : batch * (i + 1)]
        )
        assert np.array_equal(labels, lab[batch * i : batch * (i + 1)]), print(
            labels, lab[batch * i : batch * (i + 1)]
        )
