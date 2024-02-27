from qonnx.core.modelwrapper import ModelWrapper
import torch
import numpy as np

import emager_dataset
import data_processing as dp


def infer_brevitas(model: torch.nn.Module, current_inp: np.ndarray):
    brevitas_output = model(torch.from_numpy(current_inp))
    return brevitas_output.numpy()


def infer_finn_onnx(model_wrapper: ModelWrapper, current_inp: np.ndarray):
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
    finn_output = output_dict[finnonnx_out_tensor_name]
    if len(finn_output) == 1:
        return finn_output[0]
    else:
        return np.argmax(finn_output[0])


def validate_brevitas_qonnx(
    brevitas_model: torch.nn.Module,
    onnx_path: str,
    emager_path: str,
    subject: str,
    session: str,
    transform_fn,
    n_samples: int = 10,
):
    onnx_model = ModelWrapper(onnx_path)
    brevitas_model.eval()

    data, labels = emager_dataset.generate_processed_validation_data(
        emager_path, subject, session, transform_fn, save=False
    )
    data, labels = data[:n_samples], labels[:n_samples]
    data = data.astype(np.float32)

    ok, nok = 0, 0
    for i in range(n_samples):
        brevitas_output = infer_brevitas(brevitas_model, data[i])
        finn_output = infer_finn_onnx(onnx_model, data[i])

        # compare the outputs
        ok += 1 if finn_output == brevitas_output else 0
        nok += 1 if finn_output != brevitas_output else 0

        print(
            f"({i+1}/{n_samples}): OK={ok}, NOK={nok}. True label: {labels[i]}, Brevitas: {brevitas_output[0]}, Finn: {finn_output}"
        )

    return (ok / (ok + nok)) * 100
