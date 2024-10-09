from qonnx.core.modelwrapper import ModelWrapper
import torch
import numpy as np
import logging as log

from emager_py import dataset


def infer_brevitas(model: torch.nn.Module, current_inp: np.ndarray) -> np.ndarray:
    brevitas_output = model(torch.from_numpy(current_inp))
    return brevitas_output.detach().numpy()


def infer_finn_onnx(model_wrapper: ModelWrapper, current_inp: np.ndarray) -> np.ndarray:
    import finn.core.onnx_exec as oxe

    finnonnx_in_tensor_name = model_wrapper.graph.input[0].name
    finnonnx_model_in_shape = model_wrapper.get_tensor_shape(finnonnx_in_tensor_name)
    finnonnx_out_tensor_name = model_wrapper.graph.output[0].name
    # reshape to expected input (add 1 for batch dimension)
    current_inp = current_inp.reshape(finnonnx_model_in_shape)
    # create the input dictionary
    input_dict = {finnonnx_in_tensor_name: current_inp}
    # run with FINN's execute_onnx
    output_dict = oxe.execute_onnx(model_wrapper, input_dict)
    # get the output tensor
    return output_dict[finnonnx_out_tensor_name]


def validate_brevitas_qonnx(
    brevitas_model: torch.nn.Module,
    onnx_path: str,
    emager_path: str,
    subject: str,
    session: str,
    transform_fn,
    n_samples: int = 10,
    emg_shape=(4, 16),
):
    onnx_model = ModelWrapper(onnx_path)
    brevitas_model.eval()

    data, labels = dataset.generate_processed_validation_data(
        emager_path, subject, session, transform_fn, save=False
    )
    data, labels = data[:n_samples].reshape(-1, 1, *emg_shape), labels[:n_samples]
    data = data.astype(np.float32)

    ok, nok = 0, 0
    for i in range(n_samples):
        sample = data[i : i + 1]
        brevitas_output = infer_brevitas(brevitas_model, sample)
        finn_output = infer_finn_onnx(onnx_model, sample)

        log.info(f"Brevitas output shape: {brevitas_output.shape}")
        log.info(f"Finn output shape: {finn_output.shape}")

        if brevitas_output.shape[-1] > 1:
            brevitas_output = np.argmax(brevitas_output)
        else:
            brevitas_output = brevitas_output[0]

        if finn_output.shape[-1] > 1:
            finn_output = np.argmax(finn_output)
        else:
            finn_output = finn_output[0]

        # compare the outputs
        ok += 1 if finn_output == brevitas_output else 0
        nok += 1 if finn_output != brevitas_output else 0

        log.info(f"({i+1}/{n_samples}): OK={ok}, NOK={nok}")

    return (ok / (ok + nok)) * 100
