from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow import templates
import finn.builder.build_dataflow_config as build_cfg

import shutil
from shutil import copytree, make_archive
import logging as log
import os

from emager_py.finn import remote_operations as ro

CUSTOM_MODEL_PROPERTIES = {}
"""
Custom properties to be inserted into the model. Custom build steps can directly access these properties.
Otherwise, they can be accessed using `model.get_metadata_prop(key)` after using `step_insert_properties` build step.
"""


def step_insert_properties(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """
    Insert custom properties into the model, which can be accessed later in the build process.

    The properties are sourced from `CUSTOM_MODEL_PROPERTIES` dictionary.
    The keys are the property names and the values are the property values, so you must manually set them before calling this build step.

    This should be added to the build process before any property-consuming steps.
    """
    log.info("Inserting custom properties into the model: %s" % CUSTOM_MODEL_PROPERTIES)
    for key, value in CUSTOM_MODEL_PROPERTIES.items():
        model.set_metadata_prop(key, value)
    return model


def step_insert_ip_into_bd(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """
    Insert custom IP into the FINN shell Vivado project.
    This build step modifies`finn.transformation.fpgadataflow.templates.custom_zynq_shell_template` at runtime by
    deleting the line `make_wrapper ...` and inserting the TCL script in its place.

    It requires the following keys in `custom_build_steps.CUSTOM_MODEL_PROPERTIES`:
        - custom_ip_path: Path to the custom IP to be inserted into the Vivado project, which must contain some Tcl code to insert the IP into Vivado.
        - custom_ip_<VAR>: Any other custom IP-related variables that need to be replaced in the custom IP file. In the file, they must be written as `$$VAR` placeholder.

    The general workflow is to:
        - Generate the FINN block design (finn's `step_synthesize_bitfile`) which by default creates the Vivado shell project in `/tmp/finn_dev_<user>/zynq_proj_<id>`
        - Export it with `custom_build_steps.step_copy_finn_bd` OR manually copy the Vivado shell project
        - Open the project in Vivado and manually insert your custom IP, noting the Tcl commands used to do so
        - Set `CUSTOM_MODEL_PROPERTIES["custom_ip_path"] = <path_to_tcl_script>`
        - Call this build step
    """
    zynq_shell_template: list[str] = templates.custom_zynq_shell_template.splitlines()
    # idx = zynq_shell_template.index("launch_runs -to_step write_bitstream impl_1")
    for i, line in enumerate(zynq_shell_template):
        if line.startswith("make_wrapper"):
            idx = i
            zynq_shell_template.pop(i)
            break

    ip_to_insert: str = CUSTOM_MODEL_PROPERTIES["custom_ip_path"]
    with open(
        ip_to_insert,
        "r",
    ) as f:
        text = f.read()
        for key, value in CUSTOM_MODEL_PROPERTIES.items():
            if not (key.startswith("custom_ip_") and key != "custom_ip_path"):
                continue
            key = key.replace("custom_ip_", "$$")
            log.info(f"Replacing {key} with {value} in custom IP file {ip_to_insert}")
            text = text.replace(key, value)
        log.info(f"Inserting ip at custom_zynq_shell_template line {idx}")
        zynq_shell_template.insert(idx, text)

    templates.custom_zynq_shell_template = "\n".join(zynq_shell_template)

    log.info(templates.custom_zynq_shell_template)

    return model


def step_copy_finn_bd(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """
    Copy the finn-generated Vivado project to the output directory.
    Must be called after `build_dataflow_steps.step_synthesize_bitfile`
    """
    assert model.get_metadata_prop("vivado_pynq_proj")
    copytree(
        model.get_metadata_prop("vivado_pynq_proj"),
        cfg.output_dir + "vivado_zynq_proj/",
        dirs_exist_ok=True,
    )

    log.info(
        f"Vivado proj {model.get_metadata_prop('vivado_pynq_proj')} copied to {cfg.output_dir}"
    )

    return model


def step_deploy_to_pynq(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """
    Deploy the deployment package to the PYNQ board.

    Depends on `emager_pynq_path`, which is the destionation directory on the PYNQ board.
    """
    pynq_emg_path = model.get_metadata_prop("emager_pynq_path")

    shutil.rmtree(cfg.output_dir + "/deploy/finn_driver", ignore_errors=True)
    os.rename(cfg.output_dir + "/deploy/driver", cfg.output_dir + "/deploy/finn_driver")
    log.info(
        make_archive(cfg.output_dir + "/deploy", "zip", cfg.output_dir + "/deploy")
    )

    conn = ro.connect_to_pynq()
    result = conn.put(
        cfg.output_dir + "deploy.zip",
        remote=pynq_emg_path,
    )
    log.info("Uploaded {0.local} to {0.remote}".format(result))
    log.info(conn.run(f"unzip -d {pynq_emg_path} -o {pynq_emg_path}/deploy.zip"))
    log.info(conn.run(f"rm {pynq_emg_path}/deploy.zip"))
    conn.close()
    return model


def insert_custom_ip(ip_tcl_script: str, **kwargs):
    """
    Utility function to add the correct keys to add custom IP into the FINN shell project.

    Parameters:
        - ip_tcl_script: Path to the custom IP to be inserted into the Vivado project, which must contain some Tcl code to insert the IP into Vivado.
        - kwargs: Variables that are to be substituted in the Tcl script before Vivado invocation. In the Tcl script, they must be written as `$$<VAR_NAME>` placeholders.
            For example, if a Tcl script has a placeholder `$$MY_VAR`, then call this function with kwargs: MY_VAR='value'.

    Returns the updated `CUSTOM_MODEL_PROPERTIES` dictionary.
    """
    CUSTOM_MODEL_PROPERTIES["custom_ip_path"] = ip_tcl_script
    for key, value in kwargs.items():
        CUSTOM_MODEL_PROPERTIES["custom_ip_" + key] = value
    log.info(f"Updated CUSTOM_MODEL_PROPERTIES: {CUSTOM_MODEL_PROPERTIES}")
    return CUSTOM_MODEL_PROPERTIES


def default_finn_flow_export_bd() -> list:
    """
    Run the default FINN build flow and export the default FINN block design to the output directory after it is generated.
    """
    steps = build_cfg.default_build_dataflow_steps[:-2] + [
        step_copy_finn_bd,
    ]
    log.info(f"FINN Build Steps: {steps}")
    return steps


def default_finn_flow_custom_ip() -> list:
    """
    Run the default FINN build flow with custom IP insertion, block design retrieval, and deployment to PYNQ.
    """
    steps = build_cfg.default_build_dataflow_steps[:-3] + [
        step_insert_properties,
        step_insert_ip_into_bd,
        *build_cfg.default_build_dataflow_steps[-3:],
        step_copy_finn_bd,
        step_deploy_to_pynq,
    ]
    log.info(f"FINN Build Steps: {steps}")
    return steps


def finn_flow_only_deploy() -> list:
    """
    Only deploy the a pre-built deployment package to the PYNQ board.
    """
    steps = [
        step_insert_properties,
        step_deploy_to_pynq,
    ]
    log.info(f"FINN Build Steps: {steps}")
    return steps
