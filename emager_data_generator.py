import numpy as np
import redis
import time
import threading
import logging as log

import emager_utils
import emager_dataset as ed
import data_processing as dp


class EmagerDataGenerator:
    def __init__(
        self, host: str, dataset_root: str, shuffle: bool = True, n_repeats: int = 10
    ):
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
            - n_repeats: how many times to repeat serving the data (useful when == epochs)
        """
        self.__dataset_root = dataset_root
        self.r = redis.Redis(host)

        self.emg = np.ndarray((0, 64))
        self.labels = np.ndarray((1))

        self.shuffle = shuffle
        self.repeats = n_repeats

        self.update_params()

    def prepare_data(self, subject: str, session: str):
        """
        Load data from disk and prepare it for serving.

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
            emg, labels = dp.shuffle_dataset(emg, labels, self.__batch)

        self.emg = emg.astype(np.int16)
        self.labels = labels.astype(np.uint8)

        self.r.set(emager_utils.GENERATED_SAMPLES_KEY, len(self))

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
        for i in range(len(self) // self.__batch):
            emg = self.emg[self.__batch * i : self.__batch * (i + 1), :]
            labels = self.labels[self.__batch * i : self.__batch * (i + 1)]
            yield emg, labels

    def serve_data(self) -> bool:
        """
        For loop over `self.generate_data` which pushes 1 data batch to `self.__redis` in a timely manner.
        For a threaded usage, prefer `self.get_serve_thread` instead.

        Returns True when serving is finished.
        """
        lpush_time = self.__batch / self.__sampling_rate
        log.info(f"Serving data every {lpush_time:.4f} s")

        # start_time = time.perf_counter()
        for i in range(self.repeats):
            # rep_start_time = time.perf_counter()
            for emg, label in self.generate_data():
                t0 = time.perf_counter()
                p = self.r.pipeline()
                p.lpush(emager_utils.SAMPLES_FIFO_NAME, emg.tobytes())
                p.lpush(emager_utils.LABELS_FIFO_NAME, label.tobytes())
                p.execute()
                dt = time.perf_counter() - t0
                if dt < lpush_time:
                    time.sleep(lpush_time - dt)
            # log.info(
            #    f"Finished serving repeat {i} in {time.perf_counter()-rep_start_time:.3f} s"
            # )
        # log.info(
        #    f"Finished serving {self.repeats} times in {time.perf_counter()-start_time:.3f} s"
        # )
        return True

    def clear_data(self):
        """
        Clear Redis Samples and Labels FIFOs.
        """
        self.r.delete(emager_utils.SAMPLES_FIFO_NAME)
        self.r.delete(emager_utils.LABELS_FIFO_NAME)

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
        try:
            self.__batch = int(self.r.get(emager_utils.BATCH_KEY))
            self.__sampling_rate = int(self.r.get(emager_utils.FS_KEY))

            log.info(
                f"Parameters updated from Redis: batch size {self.__batch}, fs {self.__sampling_rate}"
            )
        except Exception as err:
            log.warning(f"{err} : Settings set to sensible defaults.")
            self.r.set(emager_utils.BATCH_KEY, 150)
            self.r.set(emager_utils.FS_KEY, 1000)
            self.update_params()

    def __len__(self):
        return len(self.labels)


def main():
    emager_utils.set_logging()

    r = redis.Redis()
    r.flushall()

    time.sleep(1)

    host = "localhost"
    dg = EmagerDataGenerator(host, emager_utils.DATASETS_ROOT + "EMAGER/", True)
    emg, lab = dg.prepare_data("004", "001")
    dg.get_serve_thread().start()

    time.sleep(3)

    print("Len of generated data: ", r.get(emager_utils.GENERATED_SAMPLES_KEY))
    for i in range(len(lab)):
        data = r.rpop(emager_utils.SAMPLES_FIFO_NAME)
        labels = r.rpop(emager_utils.LABELS_FIFO_NAME)

        dec = np.frombuffer(data, dtype=np.int16).reshape((-1, 64))
        labels = np.frombuffer(labels, dtype=np.uint8)

        batch = len(labels)
        print(labels)
        print(f"Received shape {dec.shape}, full data shape {emg.shape}")
        assert np.array_equal(dec, emg[batch * i : batch * (i + 1)]), print(
            dec, emg[batch * i : batch * (i + 1)]
        )
        assert np.array_equal(labels, lab[batch * i : batch * (i + 1)]), print(
            labels, lab[batch * i : batch * (i + 1)]
        )
        time.sleep(2)


if __name__ == "__main__":
    # Server's job is to deliver data to Redis
    main()
