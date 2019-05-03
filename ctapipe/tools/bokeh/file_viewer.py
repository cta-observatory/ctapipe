import os
from bokeh.layouts import widgetbox, layout
from bokeh.models import Select, TextInput, PreText, Button
from bokeh.server.server import Server
from bokeh.document.document import jinja2
from bokeh.themes import Theme
from traitlets import Dict, List, Int, Bool
from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io import EventSource
from ctapipe.io.eventseeker import EventSeeker
from ctapipe.plotting.bokeh_event_viewer import BokehEventViewer
from ctapipe.utils import get_dataset_path
import ctapipe.utils.tools as tool_utils


class BokehFileViewer(Tool):
    name = "BokehFileViewer"
    description = ("Interactively explore an event file using the bokeh "
                   "visualisation package")

    port = Int(5006, help="Port to open bokeh server onto").tag(config=True)
    disable_server = Bool(False, help="Do not start the bokeh server "
                                      "(useful for testing)").tag(config=True)

    default_url = get_dataset_path("gamma_test_large.simtel.gz")
    EventSource.input_url.default_value = default_url

    extractor_product = tool_utils.enum_trait(
        ImageExtractor,
        default='NeighborPeakWindowSum'
    )

    aliases = Dict(dict(
        port='BokehFileViewer.port',
        disable_server='BokehFileViewer.disable_server',
        f='EventSource.input_url',
        max_events='EventSource.max_events',
        extractor='BokehFileViewer.extractor_product',
    ))

    classes = List(
        [
            EventSource,
        ] + tool_utils.classes_with_traits(ImageExtractor)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event = None
        self._event_index = None
        self._event_id = None
        self._telid = None
        self._channel = None

        self.w_next_event = None
        self.w_previous_event = None
        self.w_event_index = None
        self.w_event_id = None
        self.w_goto_event_index = None
        self.w_goto_event_id = None
        self.w_telid = None
        self.w_channel = None
        self.w_dl1_dict = None
        self.wb_extractor = None
        self.layout = None

        self.reader = None
        self.seeker = None
        self.extractor = None
        self.calibrator = None
        self.viewer = None

        self._updating_dl1 = False

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"

        self.reader = EventSource.from_config(parent=self)
        self.seeker = EventSeeker(self.reader, parent=self)

        self.extractor = ImageExtractor.from_name(
            self.extractor_product,
            parent=self
        )
        self.calibrator = CameraCalibrator(
            parent=self,
            image_extractor=self.extractor,
        )

        self.viewer = BokehEventViewer(parent=self)

        # Setup widgets
        self.viewer.create()
        self.viewer.enable_automatic_index_increment()
        self.create_previous_event_widget()
        self.create_next_event_widget()
        self.create_event_index_widget()
        self.create_goto_event_index_widget()
        self.create_event_id_widget()
        self.create_goto_event_id_widget()
        self.create_telid_widget()
        self.create_channel_widget()
        self.create_dl1_widgets()
        self.update_dl1_widget_values()

        # Setup layout
        self.layout = layout([
            [self.viewer.layout],
            [
                self.w_previous_event,
                self.w_next_event,
                self.w_goto_event_index,
                self.w_goto_event_id
            ],
            [self.w_event_index, self.w_event_id],
            [self.w_telid, self.w_channel],
            [self.wb_extractor]
        ])

    def start(self):
        self.event_index = 0

    def finish(self):
        if not self.disable_server:
            def modify_doc(doc):
                doc.add_root(self.layout)
                doc.title = self.name

                directory = os.path.abspath(os.path.dirname(__file__))
                theme_path = os.path.join(directory, "theme.yaml")
                template_path = os.path.join(directory, "templates")
                doc.theme = Theme(filename=theme_path)
                env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(template_path)
                )
                doc.template = env.get_template('index.html')

            self.log.info('Opening Bokeh application on '
                          'http://localhost:{}/'.format(self.port))
            server = Server({'/': modify_doc}, num_procs=1, port=self.port)
            server.start()
            server.io_loop.add_callback(server.show, "/")
            server.io_loop.start()

    @property
    def event_index(self):
        return self._event_index

    @event_index.setter
    def event_index(self, val):
        try:
            self.event = self.seeker[val]
        except IndexError:
            self.log.warning(f"Event Index {val} does not exist")

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, val):
        try:
            self.event = self.seeker[str(val)]
        except IndexError:
            self.log.warning(f"Event ID {val} does not exist")

    @property
    def telid(self):
        return self._telid

    @telid.setter
    def telid(self, val):
        self.channel = 0
        tels = list(self.event.r0.tels_with_data)
        if val not in tels:
            val = tels[0]
        self._telid = val
        self.viewer.telid = val
        self.update_telid_widget()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, val):
        self._channel = val
        self.viewer.channel = val
        self.update_channel_widget()

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, val):
        self.calibrator(val)

        self._event = val

        self.viewer.event = val

        self._event_index = val.count
        self._event_id = val.r0.event_id
        self.update_event_index_widget()
        self.update_event_id_widget()

        self._telid = self.viewer.telid
        self.update_telid_widget()

        self._channel = self.viewer.channel
        self.update_channel_widget()

    def update_dl1_calibrator(self, extractor=None):
        """
        Recreate the dl1 calibrator with the specified extractor and cleaner

        Parameters
        ----------
        extractor : ctapipe.image.extractor.ImageExtractor
        """
        if extractor is None:
            extractor = self.calibrator.image_extractor

        self.extractor = extractor

        self.calibrator = CameraCalibrator(
            parent=self,
            image_extractor=self.extractor,
        )
        self.viewer.refresh()

    def create_next_event_widget(self):
        self.w_next_event = Button(label=">", button_type="default", width=50)
        self.w_next_event.on_click(self.on_next_event_widget_click)

    def on_next_event_widget_click(self):
        self.event_index += 1

    def create_previous_event_widget(self):
        self.w_previous_event = Button(
            label="<",
            button_type="default",
            width=50
        )
        self.w_previous_event.on_click(self.on_previous_event_widget_click)

    def on_previous_event_widget_click(self):
        self.event_index -= 1

    def create_event_index_widget(self):
        self.w_event_index = TextInput(title="Event Index:", value='')

    def update_event_index_widget(self):
        if self.w_event_index:
            self.w_event_index.value = str(self.event_index)

    def create_event_id_widget(self):
        self.w_event_id = TextInput(title="Event ID:", value='')

    def update_event_id_widget(self):
        if self.w_event_id:
            self.w_event_id.value = str(self.event_id)

    def create_goto_event_index_widget(self):
        self.w_goto_event_index = Button(
            label="GOTO Index",
            button_type="default",
            width=100
        )
        self.w_goto_event_index.on_click(self.on_goto_event_index_widget_click)

    def on_goto_event_index_widget_click(self):
        self.event_index = int(self.w_event_index.value)

    def create_goto_event_id_widget(self):
        self.w_goto_event_id = Button(
            label="GOTO ID",
            button_type="default",
            width=70
        )
        self.w_goto_event_id.on_click(self.on_goto_event_id_widget_click)

    def on_goto_event_id_widget_click(self):
        self.event_id = int(self.w_event_id.value)

    def create_telid_widget(self):
        self.w_telid = Select(title="Telescope:", value="", options=[])
        self.w_telid.on_change('value', self.on_telid_widget_change)

    def update_telid_widget(self):
        if self.w_telid:
            tels = [str(t) for t in self.event.r0.tels_with_data]
            self.w_telid.options = tels
            self.w_telid.value = str(self.telid)

    def on_telid_widget_change(self, _, __, ___):
        if self.telid != int(self.w_telid.value):
            self.telid = int(self.w_telid.value)

    def create_channel_widget(self):
        self.w_channel = Select(title="Channel:", value="", options=[])
        self.w_channel.on_change('value', self.on_channel_widget_change)

    def update_channel_widget(self):
        if self.w_channel:
            try:
                n_chan = self.event.r0.tel[self.telid].waveform.shape[0]
            except AttributeError:
                n_chan = 1
            channels = [str(c) for c in range(n_chan)]
            self.w_channel.options = channels
            self.w_channel.value = str(self.channel)

    def on_channel_widget_change(self, _, __, ___):
        if self.channel != int(self.w_channel.value):
            self.channel = int(self.w_channel.value)

    def create_dl1_widgets(self):
        self.w_dl1_dict = dict(
            extractor=Select(title="Extractor:", value='', width=5,
                             options=BokehFileViewer.extractor_product.values),
            extractor_window_start=TextInput(title="Window Start:", value=''),
            extractor_window_width=TextInput(title="Window Width:", value=''),
            extractor_window_shift=TextInput(title="Window Shift:", value=''),
            extractor_lwt=TextInput(title="Local Pixel Weight:", value=''))

        for val in self.w_dl1_dict.values():
            val.on_change('value', self.on_dl1_widget_change)

        self.wb_extractor = widgetbox(
            PreText(text="Charge Extractor Configuration"),
            self.w_dl1_dict['extractor'],
            self.w_dl1_dict['extractor_window_start'],
            self.w_dl1_dict['extractor_window_width'],
            self.w_dl1_dict['extractor_window_shift'],
            self.w_dl1_dict['extractor_lwt'])

    def update_dl1_widget_values(self):
        if self.w_dl1_dict:
            for key, val in self.w_dl1_dict.items():
                if 'extractor' in key:
                    if key == 'extractor':
                        val.value = self.extractor.__class__.__name__
                    else:
                        key = key.replace("extractor_", "")
                        try:
                            val.value = str(getattr(self.extractor, key))
                        except AttributeError:
                            val.value = ''

    def on_dl1_widget_change(self, _, __, ___):
        if self.event:
            if not self._updating_dl1:
                self._updating_dl1 = True
                cmdline = []
                for key, val in self.w_dl1_dict.items():
                    k = key.replace("extractor_", "ImageExtractor.")
                    if val.value:
                        cmdline.append(f'--{k}={val.value}')
                self.parse_command_line(cmdline)
                extractor = ImageExtractor.from_name(
                    self.extractor_product,
                    parent=self)
                self.update_dl1_calibrator(extractor)
                self.update_dl1_widget_values()
                self._updating_dl1 = False


def main():
    exe = BokehFileViewer()
    exe.run()


if __name__ == '__main__':
    main()
