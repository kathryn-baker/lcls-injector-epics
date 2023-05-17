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
        default_vals = []
        for input_name in self.features:
            if input_name in self.input_variables.keys():
                default_vals.append(self.input_variables[input_name].default)
            else:
                xrms = self.input_variables["CAMR:IN20:186:XRMS"]
                yrms = self.input_variables["CAMR:IN20:186:YRMS"]
                default_vals.append(
                    r_dist(torch.tensor(xrms.default), torch.tensor(yrms.default))
                )
        self.default_values = torch.tensor(
            default_vals, dtype=torch.double, requires_grad=True
        )

    def evaluate(self, input_dict):
        return super().evaluate(input_dict)

    def _prepare_inputs(self, input_variables):
        model_vals = super()._prepare_inputs(input_variables)

        xrms = model_vals.pop("CAMR:IN20:186:XRMS")
        yrms = model_vals.pop("CAMR:IN20:186:YRMS")
        model_vals["CAMR:IN20:186:R_DIST"] = r_dist(xrms, yrms).double()
        return model_vals
