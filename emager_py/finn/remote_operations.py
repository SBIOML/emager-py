import numpy as np
import fabric
from invoke import Responder

import emager_py.emager_redis as er

DEFAULT_EMAGER_PYNQ_PATH = "/home/xilinx/workspace/emager-pynq"


def connect_to_pynq(user: str = "xilinx", hostname: str = "pynq") -> fabric.Connection:
    """
    Return a connection handle to the PYNQ board.
    """
    return fabric.Connection(
        f"{user}@{hostname}", connect_kwargs={"password": "xilinx"}
    )


def run_remote_finn(conn: fabric.Connection, path: str, script: str) -> fabric.Result:
    """
    Run a remote Python script on PYNQ. On PYNQ, the ran script is `run.sh`,
    which takes in an arbitrary number of argument(s).

    If there are some PYNQ errors, maybe some more stuff needs to be sourced from PYNQ's `/etc/profile.d/`.

    Example: `self.run_remote_finn(c, "validate_finn.py")` runs `bash run.sh validate_finn.py` on the remote PYNQ.
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
    repeats: int,
    path: str,
    redis_host: str = "localhost",
):
    """
    Interactive process to gather training data from PYNQ.
    """
    r = er.EmagerRedis(redis_host)
    for e in range(repeats):
        for g in range(n_gestures):
            input(f"({e+1}/{repeats}) Do gesture #{g}")
            run_remote_finn(conn, path, f"rhd-sampler/build/rhd_sampler {redis_host}")
            for _ in range(r.r.llen(er.EmagerRedis.SAMPLES_FIFO_KEY)):
                r.lpush(r.LABELS_FIFO_KEY, np.array(g, dtype=np.uint8).tobytes())
    r.set(r.GENERATED_SAMPLES_KEY, n_samples * n_gestures)


if __name__ == "__main__":
    conn = connect_to_pynq()
    sample_live_data(conn, 5, 6, 1, DEFAULT_EMAGER_PYNQ_PATH, "localhost")
    conn.close()
