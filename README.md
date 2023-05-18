# lcls-injector-epics

An example implementation of LUME-EPICS and LUME-Model with the LCLS injector surrogate.

# Run

Firstly, create the environment defined in `environment.yml`. Then in a **linux** shell, run the following command to start the EPICS server:

```python server.py --no-calibration```

If you want to use a client to visualize the results of the model in real time, open a second linux shell and run the following command to pull up a `bokeh` dashboard in your browser:

```bokeh serve client.py --show```

## Notes

* currently the `environment.yml` uses a pip installed version of `lume-epics` from a local fork. Once the PR is accepted and the changes pulled into the main repo, this can be updated to use the conda version of the library
* Remember to change the `serve` flag and the `pvname` parameters in `configs/epics_config.yml` if you want to run on the live machine. Otherwise, new PVs will be created and served rather than reading from the live machine.
* Using the `--calibration` flag instead of `--no-calibration` will allow force the model to use decoupled input and output calibration parameters defined in `configs/calibration.json`.
* compound PVs (e.g. `CAMR:IN20:186:R_DIST`, a product of `"CAMR:IN20:186:R_DIST"` and `"CAMR:IN20:186:R_DIST"`) can be defined by overriding the `_prepare_inputs()` method of `lume_model.torch.PyTorchModel`.
  * it may be best in future to create a new `Variable` class that allows this behaviour
