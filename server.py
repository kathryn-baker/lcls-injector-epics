import argparse
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from lume_epics.epics_server import Server
from lume_epics.utils import config_from_yaml
from lume_model.torch import PyTorchModel
from lume_model.utils import variables_from_yaml

from model import PyTorchModelCompoundPV
from transformers import (
    get_calibration_transformers,
    get_pv_to_sim_transformers,
    get_sim_to_nn_transformers,
    model_info,
    pv_info,
)

parser = argparse.ArgumentParser(description="LCLS Injector Surrogate in EPICS")
parser.add_argument(
    "--calibration",
    action=argparse.BooleanOptionalAction,
    help="Passing this flag will tell the model to include a calibration parameter. To ignore calibration, use --no-calibration flag",
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.calibration)
    with open("configs/epics_config.yml", "r") as f:
        epics_config = config_from_yaml(f)

    with open("configs/lcls_pv_variables.yml", "r") as f:
        input_variables, output_variables = variables_from_yaml(f)

    features = [
        pv_info["sim_name_to_pv_name"][sim_name]
        for sim_name in model_info["model_in_list"]
    ]
    outputs = [
        pv_info["sim_name_to_pv_name"][sim_name]
        for sim_name in model_info["model_out_list"]
    ]
    input_pv_to_sim, output_pv_to_sim = get_pv_to_sim_transformers(features, outputs)
    output_indices = [
        model_info["loc_out"][pv_info["pv_name_to_sim_name"][pvname]]
        for pvname in outputs
    ]

    input_sim_to_nn, output_sim_to_nn = get_sim_to_nn_transformers(output_indices)
    input_calibration, output_calibration = get_calibration_transformers(
        args.calibration
    )

    # build the PyTorchModel and include the known calibration layers
    model_kwargs = {
        "model_file": "torch_model.pt",
        "input_variables": input_variables,
        "output_variables": output_variables,
        "input_transformers": [input_pv_to_sim, input_sim_to_nn, input_calibration],
        "output_transformers": [output_calibration, output_sim_to_nn, output_pv_to_sim],
        "feature_order": features,
        "output_order": outputs,
        "output_format": {"type": "variable"},
    }
    server = Server(
        PyTorchModelCompoundPV, epics_config=epics_config, model_kwargs=model_kwargs
    )

    # monitor = False does not loop in main thread
    server.start(monitor=True)
