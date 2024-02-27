import torch
import numpy as np
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import sys
import os
import subprocess as sp

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__) + "/../")
sys.path.append(os.path.dirname(__file__) + "/../../")

sp.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import utils.config  # noqa: E402
import transforms  # noqa: E402
import emager_dataset as ed  # noqa: E402
import graphs.models.qemager as qemager  # noqa: E402
import model_validation as validation  # noqa: E402
import model_transformations as mt  # noqa: E402
import add_finn_board  # noqa: E402
import custom_build_steps  # noqa: E402


if __name__ == "__main__":

    print("******************************************")
    print("Entering build_dataflow.py")
    print(
        """
This script will:
    - load the best torch model
    - convert it to finn-onnx
    - validate brevitas and onnx inference
    - export validation data
    - build and deploy finn accelerator
"""
    )
    print("******************************************")

    # Get the PyTorch configs
    config = utils.config.process_config("configs/qemager_exp_0.json")

    # Create the required paths
    build_dir = "experiments/" + config.exp_name + "/"
    onnx_model = build_dir + "qemager.onnx"

    model_f = torch.load(config.checkpoint_dir + "model_best.pth.tar")
    if "state_dict" in model_f.keys():
        model_f = model_f["state_dict"]

    # Convert brevitas to finn-onnx
    transform = transforms.transforms_lut[config.transform]
    input_bits = 16 if config.transform == "default" else 8
    torch_model = qemager.QEmager(
        config.bit_width, config.input_shape, config.num_classes
    )
    torch_model.load_state_dict(model_f)

    # Add topk on model
    topk_model = mt.EmagerTopK(torch_model)
    mt.save_model_as_qonnx(
        topk_model,
        onnx_model,
        config.input_shape,
        "INT16" if input_bits == 16 else "UINT8",
        # show = True,
    )
    # Validate brevitas vs ONNX
    validation.validate_brevitas_qonnx(
        topk_model,
        onnx_model,
        config.emager_root,
        config.subject,
        config.session,
        transform,
    )

    # Generate validation data and labels
    pvd, pvl = ed.generate_processed_validation_data(
        config.emager_root,
        config.subject,
        config.session,
        transform,
        build_dir,
    )
    pred = topk_model(torch.from_numpy(pvd)).detach().numpy()
    valid_samples = 10
    np.save(build_dir + "/input", pvd[:valid_samples])
    np.save(build_dir + "/expected_output", pred[:valid_samples])

    ed.generate_raw_validation_data(
        config.emager_root,
        config.subject,
        config.session,
        transform,
        build_dir,
    )

    # Create finn board definition if necessary
    board_name = config.board_name
    if not add_finn_board.is_board_exists(board_name):
        print(f"Generating board definition for {board_name}")
        board_name = add_finn_board.add_board(
            f'{os.environ["XILINX_VIVADO"]}/data/boards/board_files/{board_name}/A.0/board.xml',
            template_board="Pynq-Z2",
        )

    # Set build steps and build config
    dataflow_steps = build_cfg.default_build_dataflow_steps[:-3] + [
        custom_build_steps.step_custom_make_bd,
        "step_make_pynq_driver",
        "step_deployment_package",
        custom_build_steps.step_custom_deploy_to_pynq,
    ]

    # Required for tcl script and custom_make_zynq_proj
    os.environ["RHD2164_SPI_FPGA_ROOT"] = os.getcwd() + "/rhd2164-spi-fpga/"
    assert os.path.exists(
        os.environ["RHD2164_SPI_FPGA_ROOT"]
    ), f"RHD2164 FPGA Block not found: {os.environ['FINN_ROOT']}/rhd2164-spi-fpga"

    # Destination path on PYNQ device to copy packaged accelerator to
    os.environ["PYNQ_PROJ_ROOT"] = config.pynq_emager_path

    cfg = build.DataflowBuildConfig(
        # steps=build_cfg.default_build_dataflow_steps,
        steps=dataflow_steps,
        # start_step=dataflow_steps[-2],
        output_dir=build_dir + "output_%s_%s/" % ("finnemager", board_name),
        mvau_wwidth_max=36,
        target_fps=1000,
        synth_clk_period_ns=10.0,
        # fpga_part="xc7z020clg400-1",
        board=board_name,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        # stitched_ip_gen_dcp=True,
        enable_build_pdb_debug=False,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            # build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        verify_steps=[
            build_cfg.VerificationStepType.STREAMLINED_PYTHON,  # TopK breaks verify steps
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ],
        save_intermediate_models=True,
        verify_input_npy=build_dir + "input.npy",
        verify_expected_output_npy=build_dir + "expected_output.npy",
    )

    # Build dataflow cfg
    # If error during step_create_dataflow_partition, check if your model is too large
    # Also test if Conv2D layers have bias=False
    build.build_dataflow_cfg(onnx_model, cfg)
