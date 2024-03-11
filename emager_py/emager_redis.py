import redis
import subprocess as sp
import numpy as np
import logging as log


class EmagerRedis:
    FS_KEY = "rhd_sample_rate"
    BATCH_KEY = "emager_samples_batch"
    TRANSFORM_KEY = "emager_transform"
    SAMPLES_FIFO_KEY = "emager_samples_fifo"
    LABELS_FIFO_KEY = "emager_labels_fifo"
    GENERATED_SAMPLES_KEY = "emager_samples_n"
    BITSTREAM_KEY = "emager_bitstream"

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

    def set(self, key, value):
        self.r.set(key, value)

    def get(self, key):
        return self.r.get(key)

    def lpush(self, key, value):
        self.r.lpush(key, value)

    def flushall(self):
        self.r.flushall()

    def clear_data(self):
        """
        Clear Redis Samples and Labels FIFOs.
        """
        self.r.delete(self.SAMPLES_FIFO_KEY)
        self.r.delete(self.LABELS_FIFO_KEY)

    def push_sample(self, samples: np.ndarray, labels: np.ndarray):
        self.r.lpush(self.SAMPLES_FIFO_KEY, samples.astype(np.int16).tobytes())
        self.r.lpush(self.LABELS_FIFO_KEY, labels.astype(np.uint8).tobytes())

    def brpoplpush_sample(self):
        dat_bytes = self.r.brpoplpush(self.SAMPLES_FIFO_KEY, self.SAMPLES_FIFO_KEY)
        labels = self.r.brpoplpush(self.LABELS_FIFO_KEY, self.LABELS_FIFO_KEY)

        emg = np.frombuffer(dat_bytes, dtype=np.int16).reshape((-1, 64))
        if len(labels) == 1:
            label = np.frombuffer(labels, dtype=np.uint8)
            labels = np.full(len(emg), label)
        else:
            labels = np.frombuffer(labels, dtype=np.uint8)

        return emg, labels

    def brpop_sample(self):
        _, dat_bytes = self.r.brpop(self.SAMPLES_FIFO_KEY)
        _, labels = self.r.brpop(self.LABELS_FIFO_KEY)

        emg = np.frombuffer(dat_bytes, dtype=np.int16).reshape((-1, 64))
        if len(labels) == 1:
            label = np.frombuffer(labels, dtype=np.uint8)
            labels = np.full(len(emg), label)
        else:
            labels = np.frombuffer(labels, dtype=np.uint8)

        return emg, labels

    def set_sampling_params(self, fs: int, batch: int):
        self.set(self.FS_KEY, fs)
        self.set(self.BATCH_KEY, batch)

    def set_pynq_params(self, bitstream: str, transform: str):
        self.set(self.BITSTREAM_KEY, bitstream)
        self.set(self.TRANSFORM_KEY, transform)

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


def start_docker_redis() -> str | None:
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
    r = EmagerRedis(get_docker_redis_ip())
