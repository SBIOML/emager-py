from qonnx.core.modelwrapper import ModelWrapper
import finn.builder.build_dataflow_config as build_cfg
from shutil import copytree, copy, make_archive
import logging as log
import os
from fabric import Connection

from emager_py.finn import custom_make_zynq_proj


def step_custom_make_bd(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """
    Re-implementation of `finn.builder.build_dataflow_steps.step_synthesize_bitfile`,
    where the accelerator BD is generated but custom user IP is added on top.
    It's currently tremendously inflexible.

    TODO maybe use model metadata to tell what tcl scripts to run.
    """
    # Build zynq project
    partition_model_dir = cfg.output_dir + "intermediate_models/kernel_partitions"
    model = model.transform(
        custom_make_zynq_proj.ZynqBuild(
            cfg.board,
            cfg.synth_clk_period_ns,
            cfg.enable_hw_debug,
            partition_model_dir=partition_model_dir,
        )
    )

    # Create dirs
    bitfile_dir = cfg.output_dir + "/bitfile"
    os.makedirs(bitfile_dir, exist_ok=True)

    # Copy outputs
    copytree(
        model.get_metadata_prop("vivado_pynq_proj"),
        cfg.output_dir + "vivado_zynq_proj/",
        dirs_exist_ok=True,
    )
    copy(model.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.bit")
    copy(model.get_metadata_prop("hw_handoff"), bitfile_dir + "/finn-accel.hwh")

    log.info(
        f"Vivado proj {model.get_metadata_prop('vivado_pynq_proj')} copied to {cfg.output_dir}"
    )

    return model


def step_custom_deploy_to_pynq(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    pynq_emg_path = "/home/xilinx/workspace/pynq-emg/ondevice"
    if "PYNQ_PROJ_ROOT" in os.environ.keys():
        pynq_emg_path = os.environ["PYNQ_PROJ_ROOT"]

    log.info(
        make_archive(cfg.output_dir + "/deploy", "zip", cfg.output_dir + "/deploy")
    )

    with Connection(
        "xilinx@pynq",
        connect_kwargs={"password": "xilinx"},
    ) as c:
        result = c.put(
            cfg.output_dir + "deploy.zip",
            remote=pynq_emg_path,
        )
        log.info("Uploaded {0.local} to {0.remote}".format(result))
        log.info(c.run(f"unzip -d {pynq_emg_path} -o {pynq_emg_path}/deploy.zip"))
        log.info(c.run(f"rm {pynq_emg_path}/deploy.zip"))

    return model
