from lume_model.torch import PyTorchModel
from pprint import pprint


class DebuggingPyTorchModel(PyTorchModel):
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
        # print(self._output_format)

    def evaluate(self, input_dict):
        # pprint(input_dict)
        return super().evaluate(input_dict)
