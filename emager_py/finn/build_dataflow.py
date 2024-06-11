if __name__ == "__main__":
    import subprocess as sp
    import os
    import sys

    sys.path.append(os.getcwd())

    sp.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ]
    )
    sp.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"file://{os.getcwd()}/emager-py",
        ]
    )

    import torch
    import numpy as np
    import utils.config
    import graphs.models.qemager as qemager

    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as build_cfg

    from emager_py import dataset, transforms
    from emager_py.finn import (
        custom_build_steps,
        model_transformations,
        model_validation,
        boards,
    )

    # Get the PyTorch configs
    exp_name = None
    with open("current_experiment.txt", "r") as f:
        exp_name = f.read().strip()

    if exp_name is None:
        exit("No experiment name found in current_experiment.txt")

    config = utils.config.process_config(f"configs/{exp_name}.json")

    # Create the required paths
    build_dir = "experiments/" + config.exp_name + "/"
    onnx_model = build_dir + "qemager.onnx"

    model_f = torch.load(config.checkpoint_dir + "model_best.pth.tar")
    if "state_dict" in model_f.keys():
        model_f = model_f["state_dict"]

    # Convert brevitas to finn-onnx
    torch_model = qemager.QEmager(
        config.bit_width, config.input_shape, config.num_classes
    )
    torch_model.load_state_dict(model_f)

    # Add topk on model
    input_bits = 16 if config.transform == "default" else 8
    model_transformations.save_model_as_qonnx(
        torch_model,
        onnx_model,
        config.input_shape,
        "INT16" if input_bits == 16 else "UINT8",
        # show = True,
    )

    transform = transforms.transforms_lut[config.transform]

    # Validate brevitas vs ONNX
    model_validation.validate_brevitas_qonnx(
        torch_model,
        onnx_model,
        config.emager_root,
        config.subject,
        config.session,
        transform,
        10,
    )

    # Generate validation data and labels
    pvd, pvl = dataset.generate_processed_validation_data(
        config.emager_root,
        config.subject,
        config.session,
        transform,
        build_dir,
    )

    pred = torch_model(torch.from_numpy(pvd)).detach().numpy()
    valid_samples = 10
    np.save(build_dir + "/input", pvd[:valid_samples])
    np.save(build_dir + "/expected_output", pred[:valid_samples])

    dataset.generate_raw_validation_data(
        config.emager_root,
        config.subject,
        config.session,
        transform,
        build_dir,
    )

    # Create finn board definition if necessary
    board_name = config.board_name
    if not boards.is_board_exists(board_name):
        print(f"Generating board definition for {board_name}")
        board_name = boards.add_board(
            f'{os.environ["XILINX_VIVADO"]}/data/boards/board_files/{board_name}/A.0/board.xml',
            template_board="Pynq-Z2",
        )

    # Set required model metadata properties
    custom_build_steps.CUSTOM_MODEL_PROPERTIES["emager_pynq_path"] = (
        config.pynq_emager_path
    )
    custom_build_steps.insert_custom_ip(
        os.getcwd() + "/utils/insert_rhd2164.tcl",
        RHD2164_SPI_ROOT=os.getcwd() + "/rhd2164-spi-fpga/",
    )

    # Set build steps and build config
    dataflow_steps = custom_build_steps.default_finn_flow_custom_ip()
    # dataflow_steps = custom_build_steps.default_finn_flow_export_bd()
    # dataflow_steps = build_cfg.default_build_dataflow_steps

    cfg = build.DataflowBuildConfig(
        steps=dataflow_steps,
        # start_step=dataflow_steps[-8],  # Only build custom BD
        # start_step=dataflow_steps[-2], # Only copy BD
        output_dir=build_dir + "output_%s_%s/" % ("finnemager", board_name),
        mvau_wwidth_max=36,
        target_fps=100000,
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
            build_cfg.DataflowOutputType.BITFILE,
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
