import numpy as np
import time
import threading
import logging as log

from emager_py import streamers
import emager_py.utils as utils
import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.emager_redis as er


class EmagerDataGenerator:
    def __init__(
        self,
        streamer: streamers.EmagerStreamerInterface,
        dataset_root: str,
        sampling_rate: int = 1000,
        batch_size: int = 1,
        shuffle: bool = True,
    ):
        """
        This class allows you to simulate a live sampling process. It does not apply any SigProc.
        It is meant to run on a host PC, not on an embedded device.

        To sync the parameters such as sample generation rate and batch size, call `update_params()`.

        First, `prepare_data` to load some data.

        Then, call `DataGenerator.serve_data(...)` to start serving data,
        or `get_serve_thread(...)` to get a server thread which can then be `start()`ed.

        Parameters:
            - streamer: a subclass of EmagerStreamerInterface. It should implement a `write()` method.
            - dataset_root: EMaGer dataset root
            - shuffle: shuffle the data before serving
        """
        self.__dataset_root = dataset_root

        self.streamer = streamer
        self.batch = batch_size
        self.sampling_rate = sampling_rate
        self.shuffle = shuffle

        streamer.clear()

        self.emg = np.ndarray((0, 64))
        self.labels = np.ndarray((1))

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
            emg, labels = dp.shuffle_dataset(emg, labels, self.batch)

        self.emg = emg.astype(np.int16)
        self.labels = labels.astype(np.uint8)

        self.streamer.set_len(len(self.labels))

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
        lpush_time = self.batch / self.sampling_rate
        log.info(f"Serving data every {lpush_time:.4f} s")
        # rep_start_time = time.perf_counter()
        for emg, label in self.generate_data():
            t0 = time.perf_counter()
            self.streamer.write(emg, label)
            dt = time.perf_counter() - t0
            if dt < lpush_time:
                time.sleep(lpush_time - dt)
            # print(time.perf_counter() - t0)

    def get_serve_thread(self):
        """
        Return a Thread which is ready to start serving data.

        Call `t.start()` on this function's return value to start serving.
        """
        return threading.Thread(target=self.serve_data)

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    utils.set_logging()

    batch = 25
    host = er.get_docker_redis_ip()
    # host = "pynq"

    server_stream = streamers.RedisStreamer(host, True)
    dg = EmagerDataGenerator(
        server_stream, utils.DATASETS_ROOT + "EMAGER/", 100000000, batch, True
    )
    emg, lab = dg.prepare_data("004", "001")
    dg.serve_data()
    print("Len of generated data: ", len(dg))
    for i in range(len(lab)):
        # data, labels = r.brpop_sample()
        data, labels = server_stream.read()
        batch = len(labels)
        print(f"Received shape {data.shape}")
        assert np.array_equal(data, emg[batch * i : batch * (i + 1)]), print(
            "Data does not match."
        )
        assert np.array_equal(labels, lab[batch * i : batch * (i + 1)]), print(
            labels, lab[batch * i : batch * (i + 1)]
        )
