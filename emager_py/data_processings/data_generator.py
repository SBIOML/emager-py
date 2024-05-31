import numpy as np
import time
import threading
import logging as log

from emager_py import streamers
import emager_py.utils as utils
import emager_py.data_processings.dataset as ed
import emager_py.data_processings.data_processing as dp
import emager_py.utils.emager_redis as er


class EmagerDataGenerator:
    def __init__(
        self,
        streamer: streamers.EmagerStreamerInterface,
        dataset_root: str,
        sampling_rate: int = 1000,
        batch_size: int = 1,
        shuffle: bool = True,
        threaded: bool = False,
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
        self.__threaded = threaded
        self.__idx = 0

        streamer.clear()

        self.emg = np.ndarray((0, 64))
        self.labels = np.ndarray((1))

    def prepare_data(self, subject: str, session: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data from disk and prepare it for serving, including shuffling if `self.shuffle` is True.
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

        self.streamer.set_len(len(labels))
        self.__idx = 0

        self.emg = emg.reshape((len(emg) // self.batch, self.batch, 64)).astype(
            np.int16
        )
        self.labels = labels.reshape((len(labels) // self.batch, self.batch, 1)).astype(
            np.uint8
        )

        log.info(
            f"Prepared data of shape {self.emg.shape}, labels {self.labels.shape}."
        )
        return self.emg, self.labels

    def start(self):
        """
        Start serving data deamon thread.
        """
        t = threading.Thread(target=self.serve_data,  daemon=True)
        t.start()
        return t

    def serve_data(self, threaded: bool = False):
        """
        For loop over `self.generate_data` which pushes 1 data batch to `self.__redis` in a timely manner.
        For a threaded usage, prefer `self.get_serve_thread` instead.
        """
        push_ts = self.batch / self.sampling_rate
        log.info(
            f"Serving {len(self)} batches of {self.batch} elements every {push_ts:.4f} s"
        )

        self.__idx = 0
        since = time.perf_counter()
        sleep_time = push_ts
        true_ts = push_ts
        while self.push_sample():
            if self.__idx % 100 == 0:
                true_ts = (time.perf_counter() - since) / (self.__idx + 1)
                err_ts = push_ts - true_ts
                sleep_time += err_ts
                # log.info(f"true avg fs: {1/true_ts:.4f} Hz, {err_ts:.6f} s error")
                
            if sleep_time > 0:
                time.sleep(sleep_time)

    def push_sample(self):
        if self.__idx >= len(self.labels):
            return False

        self.streamer.write(self.emg[self.__idx], self.labels[self.__idx])
        self.__idx += 1
        return True

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    from emager_py.utils import utils
    utils.set_logging()

    batch = 10
    host = er.get_docker_redis_ip()
    # host = "pynq"

    # server_stream = streamers.RedisStreamer(host, True)
    server_stream = streamers.TcpStreamer(4444, listen=False)
    dg = EmagerDataGenerator(
        server_stream, utils.DATASETS_ROOT + "EMAGER/", 1000, batch, True
    )
    emg, lab = dg.prepare_data("004", "001")
    dg.serve_data(True)

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
