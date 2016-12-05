
"""
Set of diagnostic plots relating to muons 
For generic use with all muon algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy import units as u
from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.image.cleaning import tailcuts_clean
from IPython import embed

from ctapipe.plotting.camera import CameraPlotter

def plot_muon_efficiency(source):

    """
    Plot the muon efficiencies
    """
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    figip,axip = plt.subplots(1,1,figsize=(10,10))
    figrw,axrw = plt.subplots(1,1,figsize=(10,10))

    mu_eff = []
    impact_param = []
    ring_width = []

    for mu_evt in source:
        if mu_evt[0] is not None and mu_evt[1] is not None:
            mu_eff.append(mu_evt[1].optical_efficiency_muon)
            impact_param.append(mu_evt[1].impact_parameter/u.m)
            ring_width.append(mu_evt[1].ring_width/u.deg)

    ax.hist(mu_eff,20)
    ax.set_xlim(0.2*min(mu_eff),1.2*max(mu_eff))
    ax.set_ylim(0.,1.2*len(mu_eff))
    ax.set_xlabel('Muon Efficiency')
    plt.figure(fig.number)

    axip.hist(impact_param,20)
    axip.set_xlim(0.2*min(impact_param),1.2*max(impact_param))
    axip.set_ylim(0.,1.2*len(impact_param))
    axip.set_xlabel('Impact Parameter (m)')
    plt.figure(figip.number)

    axrw.hist(ring_width,20)
    axrw.set_xlim(0.2*min(ring_width),1.2*max(ring_width))
    axrw.set_ylim(0.,1.2*len(ring_width))
    axrw.set_xlabel('Ring Width ($^\circ$)')
    plt.figure(figrw.number)

    plt.show()



def plot_muon_event(event, muonparams, geom_dict=None, args=None):

    if muonparams[0] is not None:

        #Plot the muon event and overlay muon parameters
        fig = plt.figure(figsize=(16, 7))
        if args.display:
            plt.show(block=False)
        pp = PdfPages(args.output_path) if args.output_path is not None else None

        colorbar = None
        colorbar2 = None

        for tel_id in event.dl0.tels_with_data:
            npads = 2
            # Only create two pads if there is timing information extracted
            # from the calibration
            ax1 = fig.add_subplot(1, npads, 1)
            plotter = CameraPlotter(event,geom_dict)
            image = event.dl1.tel[tel_id].pe_charge
            #Get geometry
            geom = None
            if geom_dict is not None and tel_id in geom_dict:
                geom = geom_dict[tel_id]
            else:
                log.debug("[calib] Guessing camera geometry")
                geom = CameraGeometry.guess(*event.inst.pixel_pos[tel_id],
                                            event.inst.optical_foclen[tel_id])
                log.debug("[calib] Camera geometry found")
                if geom_dict is not None:
                    geom_dict[tel_id] = geom
        

            tailcuts = (5.,7.)
            #Try a higher threshold for FlashCam
            if event.inst.optical_foclen[tel_id] == 16.*u.m and event.dl0.tel[tel_id].num_pixels == 1764:
                tailcuts = (10.,12.)
        
            #print("Using Tail Cuts:",tailcuts)
            clean_mask = tailcuts_clean(geom,image,1,picture_thresh=tailcuts[0],boundary_thresh=tailcuts[1])


            signals = image*clean_mask

            #print("Ring Centre in Nominal Coords:",muonparams[0].ring_center_x,muonparams[0].ring_center_y)
            muon_incl = np.sqrt(muonparams[0].ring_center_x**2. + muonparams[0].ring_center_y**2.)

            muon_phi = np.arctan(muonparams[0].ring_center_y/muonparams[0].ring_center_x)

            rotr_angle = 0.*u.deg
            if event.inst.optical_foclen[tel_id] > 10.*u.m and event.dl0.tel[tel_id].num_pixels != 1764:
                rotr_angle = -200.28*u.deg
            

            #Convert to camera frame (centre & radius)
            ring_nominal = NominalFrame(x=muonparams[0].ring_center_x,y=muonparams[0].ring_center_y,array_direction=[event.mc.alt, event.mc.az ],pointing_direction=[event.mc.alt, event.mc.az ])

            #ring_camcoord = ring_nominal.transform_to(CameraFrame(None))
            ring_camcoord = ring_nominal.transform_to(CameraFrame(pointing_direction=[event.mc.alt, event.mc.az ],focal_length = event.inst.optical_foclen[tel_id], rotation=rotr_angle))
            

            centroid_rad = np.sqrt(ring_camcoord.y**2 + ring_camcoord.x**2)
            centroid = (ring_camcoord.x.value, ring_camcoord.y.value)

            ringrad_camcoord = muonparams[0].ring_radius.to(u.rad)*event.inst.optical_foclen[tel_id]*2.#But not FC?



            rot_angle = 0.*u.deg
            if event.inst.optical_foclen[tel_id] > 10.*u.m and event.dl0.tel[tel_id].num_pixels != 1764:
                rot_angle = -100.14*u.deg


            px, py = event.inst.pixel_pos[tel_id]

            camera_coord = CameraFrame(x=px,y=py,z=np.zeros(px.shape)*u.m, focal_length=event.inst.optical_foclen[tel_id],rotation=rot_angle)

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[event.mc.alt, event.mc.az ],pointing_direction=[event.mc.alt, event.mc.az ]))
            #,focal_length = event.inst.optical_foclen[tel_id])) # tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==telid]['FL'][0]*u.m))
        
            px = nom_coord.x.to(u.deg)
            py = nom_coord.y.to(u.deg)

            dist = np.sqrt(np.power(px-muonparams[0].ring_center_x,2) + np.power(py - muonparams[0].ring_center_y,2))
            ring_dist = np.abs(dist-muonparams[0].ring_radius)
            pixRmask = ring_dist < muonparams[0].ring_radius*0.4

            signals *= muonparams[1].mask

            camera1 = plotter.draw_camera(tel_id,signals,ax1)

            cmaxmin = (max(signals) - min(signals))
            if not cmaxmin:
                cmaxmin = 1.
            cmap_charge = colors.LinearSegmentedColormap.from_list(
                'cmap_c', [(0 / cmaxmin, 'darkblue'),
                           (np.abs(min(signals)) / cmaxmin, 'black'),
                           (2.0 * np.abs(min(signals)) / cmaxmin, 'blue'),
                           (2.5 * np.abs(min(signals)) / cmaxmin, 'green'),
                           (1, 'yellow')])
            camera1.pixels.set_cmap(cmap_charge)
            if not colorbar:
                camera1.add_colorbar(ax=ax1, label=" [photo-electrons]")
                colorbar = camera1.colorbar
            else:
                camera1.colorbar = colorbar
            camera1.update(True)
       
            camera1.add_ellipse(centroid,ringrad_camcoord.value,ringrad_camcoord.value,0.,0.,color="red")

            ax1.set_title("CT {} ({}) - Mean pixel charge"
                          .format(tel_id, geom_dict[tel_id].cam_id))

            if muonparams[1] is not None:
                #continue #Comment this...
                ringwidthfrac = 0.5*muonparams[1].ring_width/muonparams[0].ring_radius
                ringrad_inner = ringrad_camcoord*(1.-ringwidthfrac)
                ringrad_outer = ringrad_camcoord*(1.+ringwidthfrac)
                camera1.add_ellipse(centroid,ringrad_camcoord.value,ringrad_inner.value,0.,0.,color="magenta")
                camera1.add_ellipse(centroid,ringrad_camcoord.value,ringrad_outer.value,0.,0.,color="magenta")
                npads = 2
                ax2 = fig.add_subplot(1,npads,npads)
                pred = muonparams[1].prediction

                if(len(pred) != np.sum(muonparams[1].mask)):
                    print("Warning! Lengths do not match...len(pred)=",len(pred),"len(mask)=",np.sum(muonparams[1].mask))

                    
                #Numpy broadcasting - fill in the shape
                plotpred = np.zeros(image.shape)
                plotpred[muonparams[1].mask==True] = pred

                camera2 = plotter.draw_camera(tel_id,plotpred,ax2)

                c2maxmin = (max(plotpred) - min(plotpred))
                if not c2maxmin:
                    c2maxmin = 1.
                c2map_charge = colors.LinearSegmentedColormap.from_list(
                    'c2map_c', [(0 / c2maxmin, 'darkblue'),
                               (np.abs(min(plotpred)) / c2maxmin, 'black'),
                               (2.0 * np.abs(min(plotpred)) / c2maxmin, 'blue'),
                               (2.5 * np.abs(min(plotpred)) / c2maxmin, 'green'),
                               (1, 'yellow')])
                camera2.pixels.set_cmap(c2map_charge)
                if not colorbar2:
                    camera2.add_colorbar(ax=ax2, label=" [photo-electrons]")
                    colorbar2 = camera2.colorbar
                else:
                    camera2.colorbar = colorbar2
                camera2.update(True)
                plt.pause(0.2)

        
            #plt.pause(0.1)
            if pp is not None:
                pp.savefig(fig)
        
            plt.close()

