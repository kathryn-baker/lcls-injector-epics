import torch
from lume_model.torch import PyTorchModel


def r_dist(x, y):
    r = torch.sqrt(x**2 + y**2)
    return r


class PyTorchModelCompoundPV(PyTorchModel):
    def __init__(
        self,
        model_file,
        input_variables,
        output_variables,
        input_transformers,
        output_transformers,
        output_format,
        feature_order,
        output_order,
        default_vals: torch.Tensor,
    ):
        super().__init__(
            model_file,
            input_variables,
            output_variables,
            input_transformers,
            output_transformers,
            output_format,
            feature_order,
            output_order,
        )
        self.default_values = default_vals

    def _prepare_inputs(self, input_variables):
        """override this function to modify the dictionary for any compound PVs/features
        that we don't measure directly from the machine"""
        model_vals = super()._prepare_inputs(input_variables)

        xrms = model_vals.pop("CAMR:IN20:186:XRMS")
        yrms = model_vals.pop("CAMR:IN20:186:YRMS")
        model_vals["CAMR:IN20:186:R_DIST"] = r_dist(xrms, yrms).double()
        return model_vals
