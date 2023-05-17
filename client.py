from bokeh.io import curdoc
from bokeh import palettes
from bokeh.layouts import column, row
from bokeh.models import LinearColorMapper, Div, Button

from lume_epics.client.controller import Controller
from lume_model.utils import variables_from_yaml
from lume_epics.utils import config_from_yaml

from lume_epics.client.widgets.plots import ImagePlot, Striptool
from lume_epics.client.widgets.tables import ValueTable
from lume_epics.client.widgets.controls import build_sliders, EntryTable
from lume_epics.client.controller import Controller
from lume_epics.client.utils import render_from_yaml
from pathlib import Path

with open("configs/epics_config.yml", "r") as f:
    epics_config = config_from_yaml(f)

with open("configs/lcls_pv_variables.yml", "r") as f:
    input_variables, output_variables = variables_from_yaml(f)

# layout, callbacks = render_from_yaml(variable_path, epics_path, read_only=False)


# curdoc().add_root(layout)

# for callback in callbacks:
#     curdoc().add_periodic_callback(callback, 1000)  # callback called every second

# create controller from epics config
controller = Controller(epics_config)

# prepare as list for rendering
# define the variables that have range to make as sliders
sliding_variables = [
    input_var
    for input_var in input_variables.values()
    if input_var.value_range[0] != input_var.value_range[1]
]
input_variables = list(input_variables.values())
output_variables = list(output_variables.values())

# define the plots we want to see
sliders = build_sliders(sliding_variables, controller)
input_value_table = ValueTable(input_variables, controller)
output_value_table = ValueTable(output_variables, controller)
# striptool = Striptool(output_variables, controller)

striptools = [Striptool([variable], controller) for variable in output_variables]


title_div = Div(
    text=f"<b>LCLS ampl sum: Last  update {controller.last_update}</b>",
    style={
        "font-size": "150%",
        "color": "#3881e8",
        "text-align": "center",
        "width": "100%",
    },
)


def update_div_text():
    global controller
    title_div.text = f"<b>LCLS Injector: Last  update {controller.last_update}</b>"


def reset_slider_values():
    for slider in sliders:
        slider.reset()


slider_reset_button = Button(label="Reset")
slider_reset_button.on_click(reset_slider_values)

# render
curdoc().title = "LCLS Injector"
curdoc().add_root(
    # the first column spans the whole page
    column(
        row(column(title_div, width=600)),
        row(
            column(
                [slider_reset_button] + [slider.bokeh_slider for slider in sliders],
                width=350,
            ),
            # column(slider_reset_button, sliders, width=350),
            column(input_value_table.table, output_value_table.table, width=350),
            column(
                striptools[0].reset_button,
                striptools[0].plot,
                striptools[1].reset_button,
                striptools[1].plot,
            ),
            column(striptools[2].reset_button, striptools[2].plot),
            column(
                striptools[3].reset_button,
                striptools[3].plot,
                striptools[4].reset_button,
                striptools[4].plot,
            ),
            # column(output_value_table.table),
        ),
        # row(
        #     column(striptools[0].reset_button, striptools[0].plot),
        #     column(striptools[1].reset_button, striptools[1].plot),
        #     column(striptools[2].reset_button, striptools[2].plot),
        #     column(striptools[3].reset_button, striptools[3].plot),
        #     column(striptools[4].reset_button, striptools[4].plot),
        # ),
    ),
)

# add refresh callbacks to ensure that the values are updated
# curdoc().add_periodic_callback(image_plot.update, 1000)
for slider in sliders:
    curdoc().add_periodic_callback(slider.update, 1000)
for striptool in striptools:
    curdoc().add_periodic_callback(striptool.update, 1000)
curdoc().add_periodic_callback(update_div_text, 1000)
curdoc().add_periodic_callback(input_value_table.update, 1000)
curdoc().add_periodic_callback(output_value_table.update, 1000)
