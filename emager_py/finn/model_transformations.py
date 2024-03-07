import torch
from torch import nn
import logging as log


class EmagerTopK(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        y = torch.argmax(x, 1)
        return y


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
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    from qonnx.transformation.insert_topk import InsertTopK
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

    try:
        from brevitas.export import export_qonnx

        print("Using export_qonnx")
        model.cpu()
        export_qonnx(
            model, export_path=out_path, input_t=torch.randn(1, 1, *input_shape)
        )
        qonnx_cleanup(out_path, out_file=out_path)
    except ImportError:
        import brevitas.onnx as bo

        print("Using brevitas.onnx")
        model.cpu()
        bo.export_finn_onnx(model, (1, 1, *input_shape), export_path=out_path)

    # ModelWrapper
    model = ModelWrapper(out_path)
    model = tidy_up(model)

    # Annotate and insert topK
    model.set_tensor_datatype(model.graph.input[0].name, DataType[datatype])
    # model = model.transform(InsertTopK(k=1))
    model = tidy_up(model)
    model.save(out_path)
    log.info("Model saved to %s" % out_path)

    # Visualize if u want
    if show:
        from finn.util.visualization import showSrc, showInNetron

        showInNetron(out_path)
        print("Netron served at 172.17.0.2:8081")

    return model
