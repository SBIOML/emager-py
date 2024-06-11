import torch
from torch import nn
import logging as log


class AppendTopK(nn.Module):
    def __init__(self, model):
        """
        Add a TopK layer to the model, otherwise it remains unchanged.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return torch.argmax(x, 1)


def tidy_up(model):

    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.general import (
        GiveReadableTensorNames,
        GiveUniqueNodeNames,
        RemoveStaticGraphInputs,
    )

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def save_model_as_qonnx(
    model: nn.Module, out_path: str, input_shape: tuple, datatype: str, show=False
):
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    from qonnx.transformation.insert_topk import InsertTopK
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

    model.cpu()
    export_qonnx(model, export_path=out_path, input_t=torch.randn(1, 1, *input_shape))
    qonnx_cleanup(out_path, out_file=out_path)
    model: ModelWrapper
    model = ModelWrapper(out_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = tidy_up(model)
    model.set_tensor_datatype(model.graph.input[0].name, DataType[datatype])
    # model = model.transform(InsertTopK(k=1))
    model = tidy_up(model)
    model.save(out_path)

    log.info("Model saved to %s" % out_path)

    # Visualize if you want
    if show:
        from finn.util.visualization import showSrc, showInNetron

        showInNetron(out_path)
        print("Netron served at 172.17.0.2:8081")

    return model


def save_scnn_model_as_qonnx(
    model: nn.Module, out_path: str, input_shape: tuple, datatype: str, show=False
):
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

    model.cpu()
    export_qonnx(model, export_path=out_path, input_t=torch.randn(1, 1, *input_shape))
    qonnx_cleanup(out_path, out_file=out_path)
    model: ModelWrapper
    model = ModelWrapper(out_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = tidy_up(model)
    model.set_tensor_datatype(model.graph.input[0].name, DataType[datatype])
    model = tidy_up(model)
    model.save(out_path)

    log.info("Model saved to %s" % out_path)

    # Visualize if you want
    if show:
        from finn.util.visualization import showSrc, showInNetron

        showInNetron(out_path)
        print("Netron served at 172.17.0.2:8081")

    return model


def add_metadata_property_to_onnx(
    model_path: str, key: str, value, out_path: str | None = None
):
    from qonnx.core.modelwrapper import ModelWrapper

    model = ModelWrapper(model_path)
    model.set_metadata_prop(key, value)
    if out_path is None:
        model.save(model_path)
    else:
        model.save(out_path)
    return model
