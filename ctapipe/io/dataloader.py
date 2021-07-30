import re

from astropy.table import join, vstack
import astropy.units as u
import numpy as np
import tables
from tables import NoSuchNodeError
from traitlets import Bool
from tqdm.auto import tqdm

from ..core import Component
from ..instrument import SubarrayDescription
from . import read_table


__all__ = ["DataLoader"]


class DataLoader(Component):
    """Helper class to load and join tables from a ctapipe file."""
    
    load_dl1_parameters = Bool(
        help=(
            "Load DL1 parameters together with trigger and pointing data"
        ),
        default_value=True,
    ).tag(config=True)
    
    def __init__(self, input_file, progressbar=True, simulated=True,
                 config=None, parent=None, **kwargs):
        
        super().__init__(config=config, parent=parent, **kwargs)
        
        self.input_file = input_file
        self.subarray = SubarrayDescription.from_hdf(input_file)
        self.simulated = simulated
        self.split_mode = None

        if simulated:
            self.simshower_table = read_table(
                input_file, "/simulation/event/subarray/shower"
            )
        self.progressbar = progressbar
    
    def get_split_mode(self):
        
        with tables.open_file(self.input_file) as f:
            key = next(iter(f.root.dl1.event.telescope.parameters._v_children.keys()))
            by_tel_id = re.match(r'tel_\d+',  key) is not None
        
        if by_tel_id:
            split_mode = "tel_id"
        else:
            split_mode = "tel_type"
        
        return split_mode
    
    def get_trigger_data(self):
        
        trigger = read_table(self.input_file, "/dl1/event/telescope/trigger")
        
        return trigger
    
    def get_pointing_data(self):
        
        pointing_list = []
        
        for t in tqdm(
            self.subarray.tel_ids, desc="pointing", disable=not self.progressbar
        ):
            try:
                
                table = read_table(self.input_file,
                               f"/dl1/monitoring/telescope/pointing/tel_{t:03d}")
                
                #add tel_id to this table
                table["tel_id"] = t
                
                pointing_list.append(table)
                    
            except NoSuchNodeError:
                self.log.debug(f"Missing pointing data from tel_id #{t}")
        
        pointing = vstack(pointing_list)
        
        return pointing
    
    
    def load_images_by_tel_id(self, tel_ids, subarray_name="selected"):
        """Loads event data for telescopes in tel_ids into an Astropy Table, including
        joined Monte-Carlo event information and telescope description
        information (e.g. pos_x). tel_ids must all be of the same type, since
        the image size in the resulting table cannot be variable.

        Note this uses a lot of memory!

        Parameters
        ----------
        tel_ids: list
            list of telescope IDs to load (should all be of the same type, so
            use `subarray.get_tel_ids_for_type(type)`
        subarray_name: str, optional
            Name of subarray to return

        Returns
        -------
        Tuple[ctapipe.instrument.SubarrayDescription, astropy.table.Table]
            tuple of (subarray, events_table)

        """
        images = []
        true_images = []

        for tel_id in tqdm(
            tel_ids,
            desc=subarray_name,
            disable=not self.progressbar,
        ):
            try:
                images.append(
                    read_table(self.input_file, f"/dl1/event/telescope/images/tel_{tel_id:03d}")
                )
                
                if self.simulated:
                    true_images.append(
                        read_table(
                            self.input_file,
                            f"/simulation/event/telescope/images/tel_{tel_id:03d}",
                        )
                    )
            except NoSuchNodeError:
                self.log.debug(f"Missing reconstructed image from tel_id = {tel_id}")

        images = vstack(images)
        
        if self.simulated:
            true_images = vstack(true_images)
            table = join(images, true_images, keys=["obs_id", "event_id", "tel_id"])
            table = join(table, self.simshower_table, keys=["obs_id", "event_id"])
        else:
            table = images

        table = join(table, self.subarray.to_table(), keys=["tel_id"])
        return self.subarray.select_subarray(subarray_name, tel_ids), table
    
    
    def load_images_for_tel_type(self, tel_type):
        """
        Loads all image data for the given telescope type.

        Parameters
        ----------
        tel_type: str
            telescope description (e.g. "SST_ASTRI_ASTRICam")

        Returns
        -------
        SubarrayDescription, Table:
            tuple of (selected subarray, events table)
        """
        
        subarray_name = f"{str(tel_type)}_subarray"
        selected_tel_ids = self.subarray.get_tel_ids_for_type(str(tel_type))
        selected_subarray = self.subarray.select_subarray(selected_tel_ids, name = subarray_name)
        
        images = read_table(self.input_file,
                            f"/dl1/event/telescope/images/{str(tel_type)}")
        if self.simulated:
            true_images = read_table(self.input_file,
                                f"/simulation/event/telescope/images/{str(tel_type)}")
            table = join(images, true_images, keys=["obs_id", "event_id", "tel_id"])
            table = join(table, self.simshower_table, keys=["obs_id", "event_id"])
        else:
            table = images
        
        return selected_subarray, table

    
    def load_images_by_tel_type(self):
        """Loads all image data sets into a dict by tel_type string

        Returns
        -------
        Dict[str,SubarrayDescription], Dict[str,Table]:
           tuple of dictionaries for the subarray and image tables per tel_type string
        """
        table_dict = {}
        subarray_dict = {}
        for tel_type in tqdm(
            sorted(
                self.subarray.telescope_types,
                key=lambda t: -t.optics.equivalent_focal_length,
            ),
            desc="tel type",
        ):
            subarray, table = self.load_images_for_tel_type(tel_type)
            table_dict[str(tel_type)] = table
            subarray_dict[str(tel_type)] = subarray

        return subarray_dict, table_dict

    
    def get_simulated_parameters_by_tel_id(self):
        
        true_params_list = []

        for t in tqdm(
            self.subarray.tel_ids, desc="true_params", disable=not self.progressbar
        ):
            try:
                true_params_list.append(
                    read_table(
                        self.input_file,
                        f"/simulation/event/telescope/parameters/tel_{t:03d}",
                    )
                )
            except NoSuchNodeError:
                self.log.debug(f"Missing true parameters from tel_id #{t}")
        
        true_params = vstack(true_params_list)
        
        # rename true_params columns to prepend "true" to avoid join conflicts:
        for col in set(true_params.colnames) - {"obs_id", "tel_id", "event_id"}:
            true_params.rename_column(col, f"true_{col}")
        
        return true_params
    
    def get_simulated_parameters_by_tel_type(self, tel_type):
        
        return read_table(self.input_file,
                          f"/simulation/event/telescope/parameters/{tel_type}")

    def get_reconstructed_parameters_by_tel_id(self):
    
        reco_parameters_list = []
        
        for t in tqdm(
            self.subarray.tel_ids, desc="reco_params", disable=not self.progressbar
        ):
            try:
                reco_parameters_tel_id = read_table(
                    self.input_file, f"/dl1/event/telescope/parameters/tel_{t:03d}"
                )
                
                reco_parameters_list.append(reco_parameters_tel_id)
            except NoSuchNodeError:
                print(f"Missing reconstructed parameters from tel_id #{t}") #logging.debug(e)
        
        reco_parameters = vstack(reco_parameters_list)
        
        return reco_parameters
    
    def get_reconstructed_parameters_by_tel_type(self, tel_type):
        
        return read_table(self.input_file, f"/dl1/event/telescope/parameters/{tel_type}")
    
    def read_from_tel_id(self):
        
        trigger = self.get_trigger_data()
        
        data_list = []
        
        for t in tqdm(
            self.subarray.tel_ids, desc="dl1_data_by_tel_id", disable=not self.progressbar
        ):
            try:
                reco_parameters_tel_id = read_table(
                    self.input_file, f"/dl1/event/telescope/parameters/tel_{t:03d}"
                )
                
                reco_parameters_tel_id = join(
                    reco_parameters_tel_id, trigger, keys=["obs_id", "event_id", "tel_id"], join_type="left"
                )
                
                pointing_tel_id = read_table(
                    self.input_file, f"/dl1/monitoring/telescope/pointing/tel_{t:03d}"
                )
                reco_parameters_tel_id["pointing_azimuth"] = (
                    np.interp(
                        reco_parameters_tel_id["time"].mjd,
                        pointing_tel_id["time"].mjd,
                        pointing_tel_id["azimuth"].quantity.to_value(u.deg),
                    )
                    * u.deg
                )
                reco_parameters_tel_id["pointing_altitude"] = (
                    np.interp(
                        reco_parameters_tel_id["time"].mjd,
                        pointing_tel_id["time"].mjd,
                        pointing_tel_id["altitude"].quantity.to_value(u.deg),
                    )
                    * u.deg
                )
                
                data_list.append(reco_parameters_tel_id)
            except NoSuchNodeError:
                print(f"Missing reconstructed parameters from tel_id #{t}") #logging.debug(e)
        
        total_data = vstack(data_list)
        
        return total_data
        
        # join simulated information if present
        if self.simulated:
            
            true_parameters = self.get_simulated_parameters_by_tel_id()
            
            total_data = join(total_data, self.simshower_table, keys=["obs_id", "event_id"])
            
            total_data = join(
                total_data,
                true_parameters,
                keys=["obs_id", "event_id", "tel_id"],
                join_type="left",
                uniq_col_name="true",
            )
            
        return total_data
    
    def read_from_tel_type(self):
        
        total_data = {}

        trigger = self.get_trigger_data()
        
        for t in tqdm(
            self.subarray.telescope_types, desc="tel_type", disable=not self.progressbar
        ):
            
            tel_type = str(t)
            tel_ids = self.subarray.get_tel_ids_for_type(tel_type)
            
            total_data[tel_type] = self.get_reconstructed_parameters_by_tel_type(tel_type)
          
            # Join with tigger information

            total_data[tel_type] = join(
                    total_data[tel_type],
                    trigger[trigger["tel_id"] == tel_ids],
                    keys=["obs_id", "event_id", "tel_id"],
                    join_type="left"
                )

            # Interpolate 
            try:
                for tel_id in tel_ids:
                    pointing = read_table(
                        self.input_file, f"/dl1/monitoring/telescope/pointing/tel_{tel_id:03d}"
                    )
                    total_data[tel_type]["pointing_azimuth"] = (
                        np.interp(
                            total_data[tel_type]["time"].mjd,
                            pointing["time"].mjd,
                            pointing["azimuth"].quantity.to_value(u.deg),
                        )
                        * u.deg
                    )
                    total_data[tel_type]["pointing_altitude"] = (
                        np.interp(
                            total_data[tel_type]["time"].mjd,
                            pointing["time"].mjd,
                            pointing["altitude"].quantity.to_value(u.deg),
                        )
                        * u.deg
                    )
            except NoSuchNodeError:
                print(f"WARNING: missing pointing data from tel_id = {tel_id}") #logging.debug(e)
        
        # Add and join simulated information if available
        if self.simulated:
            
                true_parameters = self.get_simulated_parameters_by_tel_type(tel_type)
                
                # rename true_params columns to prepend "true" to avoid join conflicts:
                for col in set(true_parameters.colnames) - {"obs_id", "tel_id", "event_id"}:
                    true_parameters.rename_column(col, f"true_{col}")

                total_data[tel_type] = join(
                    total_data[tel_type],
                    true_parameters,
                    keys=["obs_id", "event_id", "tel_id"],
                    join_type="left",
                    uniq_col_name="true",
                )
            
                total_data[tel_type] = join(total_data[tel_type], self.simshower_table, keys=["obs_id", "event_id"])
        
        return total_data
    
    
    def get_images(self):
        
        self.split_mode = self.get_split_mode()
        
        if self.split_mode == "tel_id":
            
            data = self.load_images_by_tel_id()
            
        elif self.split_mode == "tel_type":
            
            data = self.load_images_by_tel_type()
        
        else:
            
            self.log.critical("Unable to identify dataset splitting mode.")
        
        return data
    
    
    def get_parameters(self):
        
        self.split_mode = self.get_split_mode()
        
        if self.split_mode == "tel_id":
            
            data = self.read_from_tel_id()
            
        elif self.split_mode == "tel_type":
            
            data = self.read_from_tel_type()
        
        else:
            
            self.log.critical("Unable to identify dataset splitting mode.")
        
        return data