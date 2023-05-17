import json

import torch
from botorch.models.transforms.input import AffineInputTransform, InputTransform

# load the transformers required for the model

with open("configs/pv_info.json", "r") as f:
    pv_info = json.load(f)
with open("configs/model_info.json", "r") as f:
    model_info = json.load(f)
with open("configs/normalization.json", "r") as f:
    norm_data = json.load(f)


class PVtoSimFactor(InputTransform, torch.nn.Module):
    def __init__(self, conversion: torch.Tensor) -> None:
        super().__init__()
        self._conversion = conversion
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = False

    def transform(self, x):
        self._conversion = self._conversion.to(x)
        return x * self._conversion

    def untransform(self, x):
        self._conversion = self._conversion.to(x)
        return x / self._conversion


class Calibration(torch.nn.Module):
    def __init__(self, scales: torch.Tensor, offsets: torch.Tensor) -> None:
        super().__init__()
        self._scales = scales
        self._offsets = offsets

    def forward(self, x):
        return self._scales * (x + self._offsets)

    def transform(self, x):
        return self.forward(x)

    def untransform(self, x):
        return self.forward(x)


def get_sim_to_nn_transformers(output_indices):
    input_scale = torch.tensor(norm_data["x_scale"], dtype=torch.double)
    input_min_val = torch.tensor(norm_data["x_min"], dtype=torch.double)
    input_sim_to_nn = AffineInputTransform(
        len(norm_data[f"x_min"]),
        1 / input_scale,
        -input_min_val / input_scale,
    )

    output_scale = torch.tensor(
        [norm_data["y_scale"][i] for i in output_indices], dtype=torch.double
    )
    output_min_val = torch.tensor(
        [norm_data["y_min"][i] for i in output_indices], dtype=torch.double
    )
    output_sim_to_nn = AffineInputTransform(
        len([norm_data["y_min"][i] for i in output_indices]),
        1 / output_scale,
        -output_min_val / output_scale,
    )

    return input_sim_to_nn, output_sim_to_nn


def get_pv_to_sim_transformers(features, outputs):
    # apply conversions
    input_pv_to_sim = PVtoSimFactor(
        torch.tensor(
            [pv_info["pv_to_sim_factor"][feature_name] for feature_name in features]
        )
    )

    # converting from mm to m for measured sigma to sim sigma, leaving the others as is
    output_pv_to_sim = PVtoSimFactor(
        torch.tensor([pv_info["pv_to_sim_factor"][output] for output in outputs])
    )
    return input_pv_to_sim, output_pv_to_sim


def get_calibration_transformers(use_calibration: bool):
    if use_calibration:
        with open("configs/calibration.json", "r") as f:
            calibration = json.load(f)
    else:
        with open("configs/no_calibration.json", "r") as f:
            calibration = json.load(f)
    input_calibration = Calibration(
        scales=torch.tensor(calibration["x_scale"]),
        offsets=torch.tensor(calibration["x_offset"]),
    )
    output_calibration = Calibration(
        scales=torch.tensor(calibration["y_scale"]),
        offsets=torch.tensor(calibration["y_offset"]),
    )
    return input_calibration, output_calibration
