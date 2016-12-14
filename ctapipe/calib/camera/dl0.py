"""
Module that applies the data volume reduction to the r1 container, and stores
it inside the dl0 container.
"""
from ctapipe.core import Component


class CameraDL0Reducer(Component):
    name = 'CameraDL0Reducer'

    def __init__(self, config, tool, reductor=None, **kwargs):
        """
        Parent class for the dl0 data volume reducers. Fills the dl0 container.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        reductor : ctapipe.calib.camera.reductors.Reductor
            The reductor to use to reduce the waveforms in the event.
            By default no data volume reduction is applied, and the dl0 samples
            will equal the r1 samples.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)
        if reductor is None:
            self.log.info("Applying no data volume reduction in the "
                          "conversion from R1 to DL0")
        self._reductor = reductor
        self._r1_empty_warn = False

    def check_r1_exists(self, event, telid):
        """
        Check that r1 data exists. If it does not, then do not change dl0.

        This ensures that if the containers were filled from a file containing
        r1 data, it is not overwritten by non-existant data.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.

        Returns
        -------
        bool
            True if r1.tel[telid].pe_samples is not None, else false.
        """
        r1 = event.r1.tel[telid].pe_samples
        if r1 is None:
            return True
        else:
            if not self._r1_empty_warn:
                self.log.warning("Encountered an event with no R1 data. "
                                 "DL0 is unchanged in this circumstance.")
                self._r1_empty_warn = True
            return False

    def reduce(self, event):
        """
        Abstract method to be defined in child class.

        Perform the conversion from raw R1 data to dl0 data
        (PE Samples -> Reduced PE Samples), and fill the dl0 container.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        tels = event.r1.tels_with_data
        for telid in tels:
            r1 = event.r1.tel[telid].pe_samples
            if self.check_r1_exists(event, telid):
                if self._reductor is None:
                    event.dl0.tel[telid].pe_samples = r1
                else:
                    reduction = self._reductor.reduce_waveforms(r1)
                    event.dl0.tel[telid].pe_samples = reduction
