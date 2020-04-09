"""
Detect and extract muon ring parameters, and write the muon ring and
intensity parameters to an output table.

The resulting output can be read e.g. using for example
`pandas.read_hdf(filename, 'muons/LSTCam')`
"""
from tqdm import tqdm

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Provenance
from ctapipe.core import Tool, ToolConfigurationError
from ctapipe.core import traits as t
from ctapipe.io import EventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.image.cleaning import TailcutsImageCleaner
from ctapipe.coordinates import TelescopeFrame, CameraFrame
from ctapipe.image.muon import MuonRingFitter, MuonIntensityFitter
import numpy as np

from astropy.coordinates import SkyCoord


class MuonAnalysis(Tool):
    name = 'ctapipe-reconstruct-muons'
    description = t.Unicode(__doc__)

    outfile = t.Unicode(
        None,
        allow_none=True,
        help='HDF5 output file name'
    ).tag(config=True)

    display = t.Bool(
        help='display the camera events', default=False
    ).tag(config=True)

    classes = t.List([
        CameraCalibrator, EventSource
    ])

    aliases = t.Dict({
        'input': 'EventSource.input_url',
        'outfile': 'MuonAnalysis.outfile',
        'max_events': 'EventSource.max_events',
        'allowed_tels': 'EventSource.allowed_tels',
    })

    def setup(self):
        self.source = self.add_component(EventSource.from_config(parent=self))
        self.calib = self.add_component(CameraCalibrator(
            parent=self,
            subarray=self.source.subarray
        ))
        self.ring_fitter = self.add_component(MuonRingFitter(
            parent=self,
        ))
        self.intensity_fitter = self.add_component(MuonIntensityFitter(
            parent=self,
            subarray=self.source.subarray,
        ))
        self.cleaning = self.add_component(
            TailcutsImageCleaner(
                parent=self,
                subarray=self.source.subarray,
            )
        )
        if self.outfile:
            self.writer = self.add_component(HDF5TableWriter(
                self.outfile, "muons", add_prefix=True
            ))
        self.pixels_in_tel_frame = {}

    def start(self):
        for event in self.source:
            self.analyze_array_event(event)

    def analyze_array_event(self, event):
        self.calib(event)
        for tel_id, dl1 in event.dl1.tel.items():
            self.log.debug(f'Processing event {event.index.event_id}, telescope {tel_id}')
            image = dl1.image
            clean_mask = self.cleaning(tel_id, image)

            if np.count_nonzero(clean_mask) <= 5:
                self.log.info(f'Skipping event {event.index.event_id}, has less then 5 pixels after cleaning')
                continue

            if tel_id not in self.pixels_in_tel_frame:
                self.pixels_in_tel_frame[tel_id] = self.pixel_to_telescope_frame(tel_id)

            pixel_coords = self.pixels_in_tel_frame[tel_id]
            x = pixel_coords.delta_az
            y = pixel_coords.delta_alt

            mask = clean_mask
            for i in range(3):
                ring = self.ring_fitter(x, y, image, mask)
                self.log.debug(
                    f'It {i}: r={ring.ring_radius:.2f}'
                    f', x={ring.ring_center_x:.2f}'
                    f', y={ring.ring_center_y:.2f}')

                dist = np.sqrt((x - ring.ring_center_x)**2 + (y - ring.ring_center_y)**2)
                mask = np.abs(dist - ring.ring_radius) / ring.ring_radius < 0.4

            # intensity_fitter does not support a mask yet, set ignored pixels to 0
            image[~mask] = 0

            result = self.intensity_fitter.fit(
                tel_id,
                ring.ring_center_x,
                ring.ring_center_y,
                ring.ring_radius,
                image,
                pedestal=1.1,
            )

            self.log.info(
                f'Muon fit: r={ring.ring_radius:.2f}'
                f', width={result.ring_width:.4f}'
                f', efficiency={result.optical_efficiency_muon:.2%}',
            )

            self.writer.write(f'tel_{tel_id}', [event.index, ring, result])

    def pixel_to_telescope_frame(self, tel_id):
        telescope = self.source.subarray.tel[tel_id]
        cam = telescope.camera.geometry
        camera_frame = CameraFrame(
            focal_length=telescope.optics.equivalent_focal_length,
            rotation=cam.cam_rotation,
        )
        cam_coords = SkyCoord(x=cam.pix_x, y=cam.pix_y, frame=camera_frame)
        return cam_coords.transform_to(TelescopeFrame())

    def finish(self):
        if self.outfile:
            Provenance().add_output_file(
                self.outfile,
                role='dl1.tel.evt.muon',
            )
            self.writer.close()


def main():
    tool = MuonAnalysis()
    tool.run()


if __name__ == '__main__':
    main()
