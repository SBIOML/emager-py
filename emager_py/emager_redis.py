import redis
import subprocess as sp
import numpy as np
import logging as log
from packaging import version


class EmagerRedis:
    FS_KEY = "rhd_sample_rate"
    AMPLIFIER_LOW_BW_KEY = "rhd_amp_f_low"
    AMPLIFIER_HI_BW_KEY = "rhd_amp_f_high"
    EN_DSP_KEY = "rhd_enable_dsp"
    FP_DSP_KEY = "rhd_amp_fp_dsp"

    BITSTREAM_KEY = "emager_bitstream"
    BATCH_KEY = "emager_samples_batch"
    SAMPLES_TO_GENERATE_KEY = "emager_samples_n"
    GENERATED_SAMPLES_KEY = "emager_samples_gen"
    TRANSFORM_KEY = "emager_transform"

    SAMPLES_FIFO_KEY = "emager_samples_fifo"
    LABELS_FIFO_KEY = "emager_labels_fifo"
    PREDICTIONS_FIFO_KEY = "emager_predictions_fifo"

    def __init__(self, hostname: str, port: int = 6379, **kwargs):
        try:
            self.r = redis.Redis(host=hostname, port=port, **kwargs)
        except ConnectionError:
            hostname = start_docker_redis()
            if hostname is None:
                raise Exception(
                    "Could not connect to Redis nor start EmagerRedis with Docker."
                )
            self.r = redis.Redis(host=hostname, port=port, **kwargs)

        ver = self.r.info()["redis_version"]
        log.info(f"Connected to Redis server {hostname} (v{ver})")

        self.is_lmove = True
        if version.parse(ver) < version.parse("6.2.0"):
            self.is_lmove = False

    def set(self, key, value):
        self.r.set(key, value)

    def get(self, key):
        return self.r.get(key)

    def push_fifo(self, key, value):
        self.r.lpush(key, value)

    def pop_fifo(self, key, timeout: int = 0):
        data_bytes = self.r.brpop(key)
        if data_bytes is None:
            return ()
        return data_bytes[1]

    def flushall(self):
        self.r.flushall()

    def clear_data(self):
        """
        Clear Redis Samples and Labels FIFOs and Generated Samples keys.
        """
        self.r.delete(self.SAMPLES_FIFO_KEY)
        self.r.delete(self.LABELS_FIFO_KEY)
        self.r.delete(self.GENERATED_SAMPLES_KEY)
        self.r.delete(self.PREDICTIONS_FIFO_KEY)

    def push_sample(self, samples: np.ndarray, labels: np.ndarray | None = None):
        self.r.lpush(self.SAMPLES_FIFO_KEY, samples.astype(np.int16).tobytes())
        if labels is not None:
            self.r.lpush(self.LABELS_FIFO_KEY, labels.astype(np.uint8).tobytes())

    def poppush_sample(self):
        dat_bytes, labels = None, None

        if self.is_lmove:
            dat_bytes = self.r.blmove(self.SAMPLES_FIFO_KEY, self.SAMPLES_FIFO_KEY, 0)
            labels = self.r.blmove(self.LABELS_FIFO_KEY, self.LABELS_FIFO_KEY, 0)
        else:
            dat_bytes = self.r.brpoplpush(self.SAMPLES_FIFO_KEY, self.SAMPLES_FIFO_KEY)
            labels = self.r.brpoplpush(self.LABELS_FIFO_KEY, self.LABELS_FIFO_KEY)

        return self.decode_labeled_data_bytes(dat_bytes, labels)

    def get_int(self, key) -> int:
        return int(self.r.get(key))

    def get_str(self, key) -> str:
        return self.get(key).decode("utf-8")

    def pop_sample(self, is_labelled: bool = False, timeout: int = 0):
        """
        Pop a sample from Redis FIFOs.
        """
        dat_bytes = self.r.brpop(self.SAMPLES_FIFO_KEY, timeout=timeout)
        if dat_bytes is None:
            return ()
        if is_labelled:
            labels = self.r.brpop(self.LABELS_FIFO_KEY, timeout=timeout)
            if labels is not None:
                return self.decode_labeled_data_bytes(dat_bytes[1], labels[1])

        return (self.decode_data_bytes(dat_bytes[1]),)

    def decode_labeled_data_bytes(self, data_bytes: bytes, label_bytes: bytes):
        data = self.decode_data_bytes(data_bytes)
        labels = self.decode_label_bytes(label_bytes)

        if len(labels) == 1:
            labels = np.full(len(labels), labels[0])

        return data, labels

    def decode_data_bytes(self, sample_bytes: bytes) -> np.ndarray:
        return np.frombuffer(sample_bytes, dtype=np.int16).reshape((-1, 64))

    def decode_label_bytes(self, label_bytes: bytes) -> np.ndarray:
        return np.frombuffer(label_bytes, dtype=np.uint8)

    def set_sampling_params(
        self, fs: int = 1000, batch: int = 25, n_samples: int = 5000
    ):
        self.set(self.FS_KEY, fs)
        self.set(self.BATCH_KEY, batch)
        self.set(self.SAMPLES_TO_GENERATE_KEY, n_samples)

    def set_rhd_sampler_params(
        self,
        low_bw: int = 15,
        hi_bw: int = 350,
        en_dsp: int = 0,
        fp_dsp: int = 20,
        bitstream: str = "",
    ):
        """
        Set RHD sampler parameters.

        Note from RHD2000 datasheet: it is good practice to set the DSP cutoff frequency fc higher than the analog amplifier lower cutoff frequency fL
        """
        self.set(self.AMPLIFIER_LOW_BW_KEY, low_bw)
        self.set(self.AMPLIFIER_HI_BW_KEY, hi_bw)
        self.set(self.EN_DSP_KEY, en_dsp)
        self.set(self.FP_DSP_KEY, fp_dsp)
        self.set(self.BITSTREAM_KEY, bitstream.encode())

    def set_pynq_params(self, transform: str):
        self.set(self.TRANSFORM_KEY, transform.encode())

    def dump_labelled_to_numpy(self, poppush: bool = True):
        """
        Dump Redis samples and labels FIFOs to numpy arrays, compatible with the `emager_py.dataset` format.

        Returns the dumped data as a numpy array of shape (n_labels, 1, n_samples, 64).
        """
        samples = {}

        sample_batches_in_fifo = self.r.llen(self.SAMPLES_FIFO_KEY)
        label_batches_in_fifo = self.r.llen(self.LABELS_FIFO_KEY)

        log.info(
            f"Sample batches in FIFO: {sample_batches_in_fifo}, labels: {label_batches_in_fifo}"
        )

        for i in range(min(sample_batches_in_fifo, label_batches_in_fifo)):
            data, labels = None, None
            if poppush:
                data, labels = self.poppush_sample()
            else:
                data, labels = self.pop_sample(is_labelled=True)
            label = str(labels[0])
            if label not in samples:
                log.info(f"Creating new label {label}")
                samples[label] = np.ndarray((0, 64), dtype=np.int16)

            samples[label] = np.vstack((samples[label], data.reshape((-1, 64))))

        minlen = min([v.shape[0] for v in samples.values()])

        return np.expand_dims(
            np.array([samples[k][:minlen] for k in sorted(samples.keys())]), axis=1
        )

    def __del__(self):
        self.r.close()


def get_docker_redis_ip() -> str:
    return (
        sp.run(
            [
                "docker",
                "inspect",
                "--format={{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                "emager-redis",
            ],
            stdout=sp.PIPE,
        )
        .stdout.decode("utf-8")
        .strip()
    )


def start_docker_redis():
    try:
        sp.check_output(
            [
                "docker",
                "run",
                "-p",
                "6379:6379",
                "-d",
                "--name",
                "emager-redis",
                "redis",
            ]
        )
        return get_docker_redis_ip()
    except sp.CalledProcessError as err:
        log.error(f"Could not start Docker Redis: {err}")
        return None


if __name__ == "__main__":
    import emager_py.finn.remote_operations as ro
    import emager_py.utils as eutils

    eutils.set_logging()

    r = EmagerRedis("pynq")
    r.clear_data()
    r.set_sampling_params(100, 50, 500)
    r.set_rhd_sampler_params(
        20, 300, 0, 15, ro.DEFAULT_EMAGER_PYNQ_PATH + "/bitfile/finn-accel.bit"
    )
    for i in range(12):
        labels = np.arange(6)
        r.push_sample(np.random.randint(0, 100, (25, 64)), labels[i % len(labels)])
    dumped = r.dump_labelled_to_numpy()
    print(dumped.shape)
