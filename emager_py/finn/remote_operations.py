import numpy as np
import fabric
from invoke import Responder

import emager_py.data_processings.emager_redis as er

DEFAULT_EMAGER_PYNQ_PATH = "/home/xilinx/workspace/emager-pynq/"


def connect_to_pynq(
    user: str = "xilinx", hostname: str = "pynq", password: str = "xilinx"
) -> fabric.Connection:
    """
    Return a connection handle to the PYNQ board. Make sure to `.close()` it after use.
    """
    return fabric.Connection(
        f"{user}@{hostname}", connect_kwargs={"password": password}
    )


def run_remote_finn(conn: fabric.Connection, path: str, script: str) -> fabric.Result:
    """
    Run a remote Python script on PYNQ. On PYNQ, the ran script is `run.sh`,
    which takes in an arbitrary number of argument(s).

    If there are some PYNQ errors, maybe some more stuff needs to be sourced from PYNQ's `/etc/profile.d/`.

    Example: `self.run_remote_finn(c, "python3 validate_finn.py")` runs `bash run.sh python3 validate_finn.py` on the remote PYNQ.
    Assumes sudo password is `xilinx`.

    Returns whatever `conn.run(...)` returns.
    """
    sudopass = Responder(
        pattern=r"\[sudo\] password for .*:",
        response="xilinx\n",
    )
    result = conn.run(
        f"bash {path}/run.sh {script}",
        pty=True,
        watchers=[sudopass],
    )
    return result


def sample_live_data(
    conn: fabric.Connection,
    n_samples: int,
    n_gestures: int,
    n_repeats: int,
    path: str,
    redis_host: str,
):
    """
    Interactive process to gather training data from PYNQ.

    `er.EmagerRedis.BATCH_KEY` must be set beforehand.

    Sets `er.EmagerRedis.GENERATED_SAMPLES_KEY`.
    """
    r = er.EmagerRedis(redis_host)
    r.set(r.SAMPLES_TO_GENERATE_KEY, n_samples)
    n_batches_per_it = n_samples // r.get_int(r.BATCH_KEY)
    for e in range(n_repeats):
        for g in range(n_gestures):
            input(f"({e+1}/{n_repeats}) Do gesture #{g}... Press Enter to start.")
            run_remote_finn(conn, path, f"rhd_sampler {redis_host}")
            for _ in range(n_batches_per_it):
                r.push_fifo(r.LABELS_FIFO_KEY, np.array(g, dtype=np.uint8).tobytes())
    n_batches_tot = r.r.llen(er.EmagerRedis.SAMPLES_FIFO_KEY)
    r.set(r.GENERATED_SAMPLES_KEY, int(n_batches_tot * n_repeats * n_gestures))


def sample_training_data(
    conn: fabric.Connection,
    redis_host: str,
    n_samples: int,
    path: str,
    gesture_id: int,
):
    r = er.EmagerRedis(redis_host)
    n_batches_per_it = n_samples // r.get_int(r.BATCH_KEY)
    run_remote_finn(conn, path, f"rhd_sampler {redis_host}")
    for _ in range(n_batches_per_it):
        r.push_fifo(r.LABELS_FIFO_KEY, np.array(gesture_id, dtype=np.uint8).tobytes())
    return n_batches_per_it


if __name__ == "__main__":
    conn = connect_to_pynq()
    r = er.EmagerRedis("pynq")
    r.clear_data()
    r.set_rhd_sampler_params(
        15, 300, 0, 1, DEFAULT_EMAGER_PYNQ_PATH + "/bitfile/finn-accel.bit"
    )
    r.set_sampling_params(1000, 50, 5000)
    # sample_live_data(conn, 5000, 2, 1, DEFAULT_EMAGER_PYNQ_PATH, "pynq")
    sample_training_data(conn, "pynq", 5000, DEFAULT_EMAGER_PYNQ_PATH, 1)
    conn.close()
