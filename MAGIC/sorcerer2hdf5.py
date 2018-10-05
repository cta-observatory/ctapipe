"""@package docstring
\file sorcerer2hdf5.py \brief Convert MAGIC calibrated images to hdf5 in the MPP format

More details (documentation to be implemented).
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
#    Authors:
#    MAGIC-LST group at MPP, Lab Saha. Mail to: mhuetten@mpp.mpg.de
###############################################################################

import sys


###############################################################################
def main(argv):
    
    help_message = \
' Script to convert MAGIC calibrated _Y_ files to hdf5. Only stereo data can be \
  converted so far and _Y_ files have to have been modified with the \
  add_pointingpos program. \n'
    
    ###########################################################################
    #    import modules:
    import os
    import getopt
    
    ###########################################################################
    #  read input variables:
    
    date = 'empty'
    runnr = 'empty'
    indir = 'empty'
    outdir = 'empty'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hd:r:i:o:',['date=','runnr=',
                                                             'indir=','outdir='])
    except getopt.GetoptError:
        print('Wrong input. The input options are:')
        print('-d or --date')
        print('-r or --runnr')
        print('-i or --indir')
        print('-o or --outdir')
        print('-h for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # help option
            print(help_message)
            sys.exit()
        elif opt in ('-d', '--date'):
            date = str(arg)
        elif opt in ('-r', '--runnr'):
            runnr = str(arg)
        elif opt in ('-i', '--indir'):
            indir = str(arg)
        elif opt in ('-o', '--outdir'):
            outdir = str(arg)
    
    if date == 'empty':
        print(' Please parse the date in the format YYYYMMDD with -d or --date\n\
 Use the -h option for further help.')
        sys.exit()
    if runnr == 'empty':
        print(' Please parse the run number with -r or --runnr\n\
 Use the -h option for further help.')
        sys.exit()
    if indir == 'empty':
        print(' Please parse directory with the input _Y_*.root files with -i or \
        --indir. So far, directory must contain both telescopes for each subrun.\n\
        Use the -h option for further help.')
        sys.exit()
    if outdir == 'empty':
        print(' Please parse the output directory with -o or --outdir\n\
 Use the -h option for further help.')
        sys.exit()
        
        
        
    convert(date, runnr, indir, outdir)

###############################################################################
def get_ROOT_array(file, tree, branch, leaf):
    
    import ROOT
    
    tfile = ROOT.TFile(file)
    ttree = getattr(tfile, tree)
    hist = ROOT.TH1D("hist","hist",100,-100,100)
    ttree.Project("hist", "%s.%s"%(branch,leaf),"","",100000000000,0)
    treeArr = ttree.GetV1()
    result = np.array(np.frombuffer(treeArr,count=int(hist.GetEntries())), dtype = np.double)
    return result

###############################################################################
def convert(date, runnr, path_in_sorcerer, filepath_out):

    import ROOT
    ROOT.gSystem.Load("$MARSSYS/libmars.so")
    import h5py
    import os
    from matplotlib import pyplot as plt
    import numpy as np
    import time
    import datetime
    import time
    import glob
    from scipy import interpolate
    
    from astropy import units as u
    from astropy.coordinates import Angle, EarthLocation
    from astropy.time import Time
    
    m1_pos = EarthLocation(lat=Angle('28:45:43.3 degrees') , lon=Angle('-17:53:24.2 degrees'), height=2200*u.m)
    m2_pos = EarthLocation(lat=Angle('28:45:40.8 degrees') , lon=Angle('-17:53:26.0 degrees'), height=2200*u.m)
    stereo_pos = EarthLocation(lat=Angle('28:45:42.462 degrees') , lon=Angle('-17:53:26.525 degrees'), height=2200*u.m)
    
    tel_coords = [m1_pos, m2_pos]
    
    # Observation data files
    filepaths_M1_sorcerer = sorted(glob.glob(path_in_sorcerer + "/%s_M1_%s*_Y_*.root"%(date,runnr)))
    filepaths_M2_sorcerer = sorted(glob.glob(path_in_sorcerer + "/%s_M2_%s*_Y_*.root"%(date,runnr)))
    
    ntels = 2
    
    nfiles = len(filepaths_M1_sorcerer)
    
    camgeom = ROOT.MGeomCamMagicTwo()
    
    camera = ROOT.MHCamera(camgeom)
    
    filepaths = [filepaths_M1_sorcerer, filepaths_M2_sorcerer]
    filename_out = ''.join(''.join(filepaths_M1_sorcerer[0].split("_M1")).split(".001"))
    filename_out = os.path.join(filepath_out,filename_out.split("/")[-1])
    filename_out = os.path.join(filepath_out,filename_out.split(".root")[0] + ".hdf5")
    
    print "Output will be written to:", filename_out
    
    arr_phe_3D = []
    arr_arrival_3D = []
    StereoEvtNumber_2D = []
    TrigPattern_2D = []
    DAQEvtNumber_2D = []
    ClockCounter_2D = []
    TimeDiff_2D = []
    MJD_2D = []
    Sec_since_MJD_2D = []
    
    Pointings_AzAlt_2D = []
    Pointings_AzAlt_corr_2D = []
    Pointings_HaDec_corr_2D = []
    Pointings_RaDec_corr_2D = []
    Pointing_RaDec = np.zeros(shape=(2,2))
    
    BadPixels_3D = []
    
    Air_temperature_2D = []
    Air_pressure_2D = []
    Air_humidity_2D = []
    
    
    # MC info:
    Energy_2D = [] # TeV
    Core_params_2D = [] # m
    X_max_2D = [] # m
    H_first_int_2D = [] # m
    Shower_angles_2D = []
    
    
    M1_pos = np.array([35., -24., 0])
    M2_pos = np.array([-35., 24., 0])
    TelPos_2D = [(35, -35), (-24, 24), (0, 0)]
    
    TrigPatternNames = []
    # valid for runs after 2013-12-13, see MARS MTriggerPatternDecode.cc, ll. 224        
    TrigPatternNames.append('L3 trigger')
    TrigPatternNames.append('Calibration')
    TrigPatternNames.append('LIDAR')
    TrigPatternNames.append('Pedestal')
    TrigPatternNames.append('Pulse injection')
    TrigPatternNames.append('Sumtrigger')
    TrigPatternNames.append('Topo trigger')
    TrigPatternNames.append('L1 trigger')
    
    n_badpixtype = 26
    
    
    nevents = np.zeros(2)
    
    for ntel in range(ntels):
    
        arr_phe_2D = []
        arr_arrival_2D = []
        arr_triggerid = []
    
        DAQEvtNumber = []
        ClockCounter = []
        NumTrigLvl2 = []
        StereoEvtNumber = []
        TriggerID = []
        TrigPattern = []
        CalibrationPattern = []
        MacrocellsPattern = []
        TimeDiff = []
        BadPixels = []
        
        Air_temperature = []
        Air_pressure = []
        Air_humidity = []
        
        MJD = []
        Sec_since_MJD = []
        
        Pointings_AzAlt = []
        Pointings_AzAlt_corr = []
        Pointings_HaDec_corr = []
        Pointings_RaDec_corr = []
        
        Air_temperature = []
        Air_pressure = []
        Air_humidity = []
        
        # MC infos
        Energy = [] # TeV
        Core_params = [] # m
        X_max = [] # m
        H_first_int = [] # m
        Shower_angles = []
        
        for nfile in range(nfiles):
            
            # get runheader information:
            Runheader = ROOT.TChain("RunHeaders")
            Runheader.Add(filepaths[ntel][nfile])
            
            header_mraw = ROOT.MRawRunHeader()
            
            header_badpixels = ROOT.MBadPixelsCam()
            Runheader.SetBranchAddress("MRawRunHeader.",header_mraw)
            Runheader.SetBranchAddress("MBadPixelsCam.",header_badpixels)
           
            # extract global information out of first event:
            Runheader.GetEntry(0)
            
            TelescopeNumber =  header_mraw.GetTelescopeNumber()
            Pointing_RaDec[ntel][0] = 90 - header_mraw.GetTelescopeRA()/3600.
            Pointing_RaDec[ntel][1] = header_mraw.GetTelescopeDEC()/3600.
    
            Source_Ra = 90 - header_mraw.GetSourceRA()/3600.
            Source_Dec = header_mraw.GetSourceDEC()/3600.
            Source_name = header_mraw.GetSourceName()
            RunNumber = header_mraw.GetRunNumber()
            RunType = header_mraw.GetRunTypeStr()
    
            if RunType == "Data":
                is_mc = False
            elif RunType == "Monte Carlo":
                is_mc = True
            else:
                is_mc = False
            
            # get weather information:
            if is_mc == False:
                Weather = ROOT.TChain("Weather")
                Weather.Add(filepaths[ntel][nfile])
                weather_report = ROOT.MReportWeather()
                Weather.SetBranchAddress("MReportWeather.",weather_report)
                weather_time = ROOT.MTime()
                Weather.SetBranchAddress("MTimeWeather.",weather_time)
    
                Air_temperature_grid = []
                Air_pressure_grid = []
                Air_humidity_grid = []
                Air_times_sec_since_mjd = []
                for weather_entry in range(Weather.GetEntries()):
                    Weather.GetEvent(weather_entry)
                    time_s = weather_time.GetTime()/1e3 + weather_time.GetNanoSec()/1e9
                    if weather_entry == 0 or (weather_entry > 0 and time_s > Air_times_sec_since_mjd[-1]):
                        Air_times_sec_since_mjd.append(time_s)
                        Air_temperature_grid.append(weather_report.GetTemperature())
                        Air_pressure_grid.append(weather_report.GetPressure())
                        Air_humidity_grid.append(weather_report.GetHumidity())
             
            # get event information:
            Events = ROOT.TChain("Events")
            Events.Add(filepaths[ntel][nfile])
            
            nevents_tmp = Events.GetEntries()
    
            evtdata_phe = ROOT.MCerPhotEvt()
            evtdata_time = ROOT.MTime()
            evtdata_arrival = ROOT.MArrivalTime()
    
            evtheader = ROOT.MRawEvtHeader()
            if is_mc == True:
                mcevtheader = ROOT.MMcEvt()
                
            Events.SetBranchAddress("MCerPhotEvt.",evtdata_phe)
            Events.SetBranchAddress("MArrivalTime.",evtdata_arrival)
            Events.SetBranchAddress("MRawEvtHeader.",evtheader)
            Events.SetBranchAddress("MTime.",evtdata_time)
            if is_mc == True:
                Events.SetBranchAddress("MMcEvt.",mcevtheader)
    
            if is_mc == False:
                evtdata_pointings = ROOT.MPointingPos()
                Events.SetBranchAddress("MPointingPos.",evtdata_pointings)
                
            start_time = time.time()
    
            for nevent in range(nevents_tmp):
    
                if nevent >= nevents_tmp:
                    continue
    
                #if nevent % 10 == 0:
                sys.stdout.write("\r ... processing event " + str(nevent) + ", "
                                  + "%s >>> Percent Done : %.0f%%" % (datetime.datetime.now(), 100*nevent/float(nevents_tmp)))
    
                Runheader.GetEvent(nevent)
                Events.GetEvent(nevent)
    
                trigid = evtheader.GetTriggerID()
    
                binarypattern = '\t{0:016b}'.format(trigid&0xff) 
    
                triggpattern = (bool(int(binarypattern[-8])), 
                                bool(int(binarypattern[-1])), 
                                bool(int(binarypattern[-2])), 
                                bool(int(binarypattern[-3])), 
                                bool(int(binarypattern[-4])), 
                                bool(int(binarypattern[-5])), 
                                bool(int(binarypattern[-6])), 
                                bool(int(binarypattern[-7])))
    
                # check mono-character of run:
                #if nevent > 0:
                 #   if triggpattern[0] == False:
                 #       if StereoEvtNumber[-1] == StereoEvtNumber[-2]:
                 #          StereoEvtNumber[-1] = -1
                        #else:
                           # print "calib!"
    
                camera.SetCamContent(evtdata_phe)
                camera.SetAllUsed()
                
                npixels = evtdata_phe.GetNumPixels()
                if npixels == None or npixels <= 0:
                    continue
                
                arr_phe_evt = []
                for i_pixel in range(npixels):
                    arr_phe_evt.append(camera.GetBinContent(i_pixel + 1))
    
                camera.SetCamContent(evtdata_arrival)
                camera.SetAllUsed()
                arr_arrival_evt = []
                for i_pixel in range(npixels):
                    arr_arrival_evt.append(camera.GetBinContent(i_pixel + 1))
        
                if arr_phe_evt != []:
                    
                    nevents[ntel] += 1
                    
                    arr_phe_2D.append(np.array(arr_phe_evt))
                    arr_arrival_2D.append(np.array(arr_arrival_evt))
    
                    DAQEvtNumber.append(evtheader.GetDAQEvtNumber())
                    ClockCounter.append(evtheader.GetClockCounter())
                    NumTrigLvl2.append(evtheader.GetNumTrigLvl2())
                    if is_mc == True:
                        nr = evtheader.GetStereoEvtNumber() 
                        if nr != 0:
                            StereoEvtNumber.append(nr + nfile * 1000)
                        else:
                            StereoEvtNumber.append(nr)
                    else:
                        StereoEvtNumber.append(evtheader.GetStereoEvtNumber())
                    TriggerID.append(evtheader.GetTriggerID())
                    TrigPattern.append(triggpattern)
                    CalibrationPattern.append(evtheader.GetCalibrationPattern())
                    MacrocellsPattern.append(evtheader.GetMacrocellsPattern())
                    TimeDiff.append(evtheader.GetTimeDiff())
                    MJD.append(np.floor(evtdata_time.GetDay()))
                    Sec_since_MJD.append(evtdata_time.GetTime()/1e3 + evtdata_time.GetNanoSec()/1e9)
                    
                    # so far, there does not seem to be difference between the bad pixels event by event
                    if BadPixels == []:
                        badpixelvals = np.zeros(shape=(n_badpixtype, npixels), dtype=np.bool)
                        for ipixel in range(npixels):
                            for i_badpixtype in range(n_badpixtype):
                                val_tmp = ROOT.Double()
                                header_badpixels.GetPixelContent(val_tmp, ipixel, camgeom, i_badpixtype)
                                badpixelvals[i_badpixtype, ipixel] = val_tmp
                        BadPixels.append(badpixelvals)
                    
                    if is_mc == False:       
                        Pointings_AzAlt.append([evtdata_pointings.GetAz(), 90 - evtdata_pointings.GetZd()])
                        Pointings_AzAlt_corr.append([evtdata_pointings.GetCorrAz(), 90 - evtdata_pointings.GetCorrZd()])
                        
                        Pointings_HaDec_corr.append([evtdata_pointings.GetCorrHa(), evtdata_pointings.GetCorrDec()])
                        time_ev = Time(MJD[-1], scale='utc', format='mjd', location=stereo_pos) + Sec_since_MJD[-1] * u.s
                        local_sidereal_time = time_ev.sidereal_time('mean')  
                        RA_corr = local_sidereal_time.deg - evtdata_pointings.GetCorrHa()  / 24. * 360
                        Pointings_RaDec_corr.append([RA_corr, evtdata_pointings.GetCorrDec()])
                                             
                        # weather information:
                        try:
                            air_temp_interpol = interpolate.interp1d(Air_times_sec_since_mjd, 
                                                                     Air_temperature_grid, fill_value='extrapolate')
                            air_press_interpol = interpolate.interp1d(Air_times_sec_since_mjd, 
                                                                     Air_pressure_grid, fill_value='extrapolate')
                            air_humid_interpol = interpolate.interp1d(Air_times_sec_since_mjd, 
                                                                     Air_humidity_grid, fill_value='extrapolate')
    
                            Air_temperature.append(air_temp_interpol(Sec_since_MJD[-1]))
                            Air_pressure.append(air_press_interpol(Sec_since_MJD[-1]))
                            Air_humidity.append(air_humid_interpol(Sec_since_MJD[-1]))
                            
                        except:
                            #if len(Air_temperature_grid) > 1 or len(Air_temperature_grid) == 0:
                            #    print "\n Warning! Something went wrong with interpolation of weather information. Length of arrays =",len(Air_temperature_grid),"\n"
                            try:
                                Air_temperature.append(Air_temperature_grid[0])
                                Air_pressure.append(Air_pressure_grid[0])
                                Air_humidity.append(Air_humidity_grid[0])
                            except:
                                Air_temperature.append(Air_temperature[-1])
                                Air_pressure.append(Air_pressure[-1])
                                Air_humidity.append(Air_humidity[-1])

                    else:
                        Pointings_AzAlt.append([np.rad2deg(mcevtheader.GetTelescopePhi()), 
                                          90. - np.rad2deg(mcevtheader.GetTelescopeTheta())])
                        Pointings_AzAlt_corr.append([np.nan, np.nan])
                        
                        energy_val = str(mcevtheader.GetEnergyStr())[:-3]
                        energy_unit = str(mcevtheader.GetEnergyStr())[-3:]
                        if energy_unit == "TeV":
                            Energy.append(float(energy_val))
                        elif energy_unit == "GeV":
                            Energy.append(float(energy_val) / 1000.)
                        else:
                            raise IOError("Error!")
                        Core_params.append((mcevtheader.GetCoreX()/100., mcevtheader.GetCoreY()/100.)) # meters
                        H_first_int.append(mcevtheader.GetZFirstInteraction()) # meters
    
            end_time = time.time()
            elapsed = end_time - start_time
            print "\n File %s read into python arrays >>> Elapsed time: %.1f s" %(filepaths[ntel][nfile], elapsed)
          
        # give MC mono events an ordered event number:
        if is_mc == True:
            dt_events = np.double
            for i_event in range(len(StereoEvtNumber)):
                if StereoEvtNumber[i_event] == 0:
                    if i_event != 0:
                        StereoEvtNumber[i_event] = np.random.uniform(StereoEvtNumber[i_event - 1], 
                                                                     np.floor(StereoEvtNumber[i_event - 1]) + 1)
                    else:
                        StereoEvtNumber[i_event] = np.random.uniform(0, 1)
        else:
            dt_events = np.int32
            
        StereoEvtNumber_2D.append(np.array(StereoEvtNumber, dtype=dt_events))
        DAQEvtNumber_2D.append(np.array(DAQEvtNumber, dtype=np.int32))
        arr_phe_3D.append(np.array(arr_phe_2D))
        arr_arrival_3D.append(np.array(arr_arrival_2D))
        TrigPattern_2D.append(TrigPattern)
        MJD_2D.append(MJD)
        Sec_since_MJD_2D.append(Sec_since_MJD)
        ClockCounter_2D.append(np.array(ClockCounter, dtype=np.int64))
        TimeDiff_2D.append(np.array(TimeDiff))
        Pointings_AzAlt_2D.append(Pointings_AzAlt)
        Pointings_AzAlt_corr_2D.append(Pointings_AzAlt_corr)
        Pointings_RaDec_corr_2D.append(Pointings_RaDec_corr)
        
        Air_temperature_2D.append(Air_temperature)
        Air_pressure_2D.append(Air_pressure)
        Air_humidity_2D.append(Air_humidity)    
        
        BadPixels_3D.append(np.array(BadPixels, dtype=np.bool))
        
        if is_mc == True:   
            Energy_2D.append(np.array(Energy))
            Core_params_2D.append(np.array(Core_params))
            #X_max_2D.append(
            H_first_int_2D.append(np.array(H_first_int))
            #Shower_angles_2D = []
    
    Pointings_AzAlt_2D = np.array(Pointings_AzAlt_2D)
    Pointings_AzAlt_corr_2D = np.array(Pointings_AzAlt_corr_2D)
    Pointings_RaDec_corr_2D = np.array(Pointings_RaDec_corr_2D)
    
    Air_temperature_2D = np.array(Air_temperature_2D)
    Air_pressure_2D = np.array(Air_pressure_2D)
    Air_humidity_2D = np.array(Air_humidity_2D)
    
    
    # In[8]:
    
    
    # Separate data and pedestal events and make stereo events match:
    
    arr_phe_3D_data = []
    arr_arrival_3D_data = []
    #arr_badpixels_4D_data = []
    tels_with_data = []
    gps_time = []
    pointing = []
    
    pointing_radec = []
    
    air_temperature = []
    air_pressure = []
    air_humidity = []
    
    # MC info
    energy = []
    h_first_int = []
    core_xy = []
    
    eventstream_corrected = [] # some stereo events are double pedestal triggers
    
    # all data events:
    eventstream = np.union1d(StereoEvtNumber_2D[0], StereoEvtNumber_2D[1])
    
    # all stereo events:
    stereoevents = np.intersect1d(StereoEvtNumber_2D[0], StereoEvtNumber_2D[1])
    nstereo = len(stereoevents)
    nstereo_check = 0
    
    arr_badpixels_3D_data = np.zeros((2,n_badpixtype,npixels,), dtype = np.bool)
    arr_badpixels_3D_data[0] = BadPixels_3D[0][0]
    arr_badpixels_3D_data[1] = BadPixels_3D[1][0]
    
    for event_id in eventstream:
    
        sys.stdout.write("\r ... processing data event " + str(event_id))
    
        stereopatterns = np.zeros(2, dtype=np.bool)
        stereotimestamps = np.zeros(shape=(2,2)) - 1.
            
        arr_phe_2D_data = np.empty((2,npixels,))
        arr_phe_2D_data[:] = None
        arr_arrival_2D_data = np.empty((2,npixels,))
        arr_arrival_2D_data[:] = None
      
        pointings = np.empty((2,2,2))
        pointings[:] = None
        pointings_radec = np.empty((2,2))
        pointings_radec[:] = None
        
        weather_info = np.empty((2,3))
        weather_info[:] = -999
        
        for ntel in range(ntels):
            
            # search event
            if event_id not in StereoEvtNumber_2D[ntel]:
                nevent = -1
            else:
                nevent = np.searchsorted(StereoEvtNumber_2D[ntel], event_id, side='left')
                
            # sort out remaining calibration runs:
            if TrigPattern_2D[ntel][nevent][0] == False:
                nevent = -1    
        
            if nevent != -1:
                arr_phe_2D_data[ntel] = arr_phe_3D[ntel][nevent,:]
                arr_arrival_2D_data[ntel] = arr_arrival_3D[ntel][nevent,:]
                stereopatterns[ntel] = True
                stereotimestamps[ntel][0] = MJD_2D[ntel][nevent]
                stereotimestamps[ntel][1] = Sec_since_MJD_2D[ntel][nevent]
      
                pointings[ntel,0,0] = Pointings_AzAlt_2D[ntel][nevent][0]
                pointings[ntel,1,0] = Pointings_AzAlt_2D[ntel][nevent][1]
                pointings[ntel,0,1] = Pointings_AzAlt_corr_2D[ntel][nevent][0]
                pointings[ntel,1,1] = Pointings_AzAlt_corr_2D[ntel][nevent][1]
                
                pointings_radec[ntel,0] = Pointings_RaDec_corr_2D[ntel][nevent][0]
                pointings_radec[ntel,1] = Pointings_RaDec_corr_2D[ntel][nevent][1]
    
                weather_info[ntel,0] = Air_temperature_2D[ntel][nevent]
                weather_info[ntel,1] = Air_pressure_2D[ntel][nevent]
                weather_info[ntel,2] = Air_humidity_2D[ntel][nevent]
    
        if (stereopatterns == [False, False]).all():
            # no stereo event, accidental double pedestal trigger
            continue
        elif (stereopatterns == [False, True]).all():
            tels_with_data.append((False, True))
        elif (stereopatterns == [True, False]).all():
            tels_with_data.append((True, False))
        elif (stereopatterns == [True, True]).all():
            tels_with_data.append((True, True))
            
        gps_time.append((stereotimestamps[0][0], stereotimestamps[0][1], stereotimestamps[1][0], stereotimestamps[1][1]))
        arr_phe_3D_data.append(arr_phe_2D_data)
        arr_arrival_3D_data.append(arr_arrival_2D_data)
        
        # so far, there does not seem to be a difference between the bad pixels event by event 
        #arr_badpixels_4D_data.append(arr_badpixels_3D_data)
        
        pointing.append((pointings[0,0,1], pointings[0,1,1], 
                         pointings[1,0,1], pointings[1,1,1]))
        
        pointing_radec.append((pointings_radec[0,0], pointings_radec[0,1],
                               pointings_radec[1,0], pointings_radec[1,1]))
    
        air_temperature.append(max(weather_info[0,0], weather_info[1,0]))
        air_pressure.append(max(weather_info[0,1], weather_info[1,1]))
        air_humidity.append(max(weather_info[0,2], weather_info[1,2]))
    
        
        eventstream_corrected.append(event_id)
        
    
    
    nevents_data = len(eventstream_corrected)
    
    energy = np.zeros(nevents_data)
    h_first_int = np.zeros(nevents_data)
    core_xy_tmp = np.zeros(shape=(nevents_data,2))
    
    
    
    event_id_pedestal = 0
    for ntel in range(ntels):
        for nevent in range(len(StereoEvtNumber_2D[ntel])):
    
            if is_mc == False:
            # loop again through all events to catch the pedestal events:
                if TrigPattern_2D[ntel][nevent][3] == True:
    
                    event_id_pedestal -= 1 # count pedestal events to negative
    
                    sys.stdout.write("\r ... processing pedestal event " + str(event_id_pedestal))
    
                    if ntel == 0:
                        tels_with_data.append((True, False))
                    else:
                        tels_with_data.append((False, True))
    
                    stereotimestamps = np.zeros(shape=(2,2)) - 1.
                    stereotimestamps[ntel][0] = MJD_2D[ntel][nevent]
                    stereotimestamps[ntel][1] = Sec_since_MJD_2D[ntel][nevent]
    
                    pointings = np.empty((2,2,2))
                    pointings[:] = None
                    pointings_radec = np.empty((2,2))
                    pointings_radec[:] = None
                              
                    pointings[ntel,0,0] = Pointings_AzAlt_2D[ntel][nevent][0]
                    pointings[ntel,1,0] = Pointings_AzAlt_2D[ntel][nevent][1]
                    pointings[ntel,0,1] = Pointings_AzAlt_corr_2D[ntel][nevent][0]
                    pointings[ntel,1,1] = Pointings_AzAlt_corr_2D[ntel][nevent][1]
                              
                    pointings_radec[ntel,0] = Pointings_RaDec_corr_2D[ntel][nevent][0]
                    pointings_radec[ntel,1] = Pointings_RaDec_corr_2D[ntel][nevent][1]
                    #arr_badpixels_3D_data[ntel] = BadPixels_3D[ntel][nevent]
    
                    air_temperature.append(Air_temperature_2D[ntel][nevent])
                    air_pressure.append(Air_pressure_2D[ntel][nevent])
                    air_humidity.append(Air_humidity_2D[ntel][nevent])
                    
                    gps_time.append((stereotimestamps[0][0], stereotimestamps[0][1], stereotimestamps[1][0], stereotimestamps[1][1]))
                    eventstream_corrected.append(event_id_pedestal)
    
                    arr_phe_2D_data = np.empty((2,npixels,))
                    arr_phe_2D_data[:] = None
                    arr_arrival_2D_data = np.empty((2,npixels,))
                    arr_arrival_2D_data[:] = None
    
                    arr_phe_2D_data[ntel] = arr_phe_3D[ntel][nevent,:]
                    arr_arrival_2D_data[ntel] = arr_arrival_3D[ntel][nevent,:]
                    arr_phe_3D_data.append(arr_phe_2D_data)
                    arr_arrival_3D_data.append(arr_arrival_2D_data)   
                    #arr_badpixels_4D_data.append(arr_badpixels_3D_data)
    
                    pointing.append((pointings[0,0,1], pointings[0,1,1], 
                                     pointings[1,0,1], pointings[1,1,1]))
                              
                    pointing_radec.append((pointings_radec[0,0], pointings_radec[0,1],
                                           pointings_radec[1,0], pointings_radec[1,1]))
                    
            elif is_mc == True:
            # collect MC information:
                event_id = StereoEvtNumber_2D[ntel][nevent]
            
                index = np.searchsorted(eventstream_corrected, event_id, side='left')
                
                energy[index] = Energy_2D[ntel][nevent]
                h_first_int[index] = H_first_int_2D[ntel][nevent]
                core_xy_tmp[index] = Core_params_2D[ntel][nevent]
        
    core_xy = []     
    for index in range(len(core_xy_tmp)):
        core_xy.append((core_xy_tmp[index,0], core_xy_tmp[index,1]))
    
    arr_phe_3D_data = np.array(arr_phe_3D_data, dtype=np.single)
    arr_arrival_3D_data = np.array(arr_arrival_3D_data, dtype=np.single)
    
    eventstream_corrected = np.array(eventstream_corrected, dtype=dt_events)
    nevents_total = len(eventstream_corrected)           
    
    
    # In[9]:
    
    
    f = h5py.File(filename_out,"w")
    
    f.attrs['Nevents_data'] = nevents_data
    f.attrs['Nevents_pedestal'] = nevents_total - nevents_data
    f.attrs['Nevents_total'] = nevents_total
    f.attrs['dl_export'] = np.string_("dl1")
    f.attrs['dl_export_comment'] = np.string_("MAGIC Sorcerer output")
    f.attrs['data format'] = np.string_("stereo")
    #f.attrs['data format'] = np.string_("mono M1")
    #f.attrs['data format'] = np.string_("mono M2")
    f.attrs['instrument'] = np.string_("MAGIC")
    f.attrs['RunType'] = "Data" #RunType
    f.attrs['RunNumber'] = RunNumber
    
    f.attrs['Observatory_lonlat_deg'] = np.array([stereo_pos.longitude.deg, stereo_pos.latitude.deg], dtype=np.single)
    f.attrs['Observatory_height_m'] = np.single(stereo_pos.height)
    
    f.attrs['SourceName'] = Source_name
    f.attrs['SourceRADEC_deg'] = np.array([Source_Ra, Source_Dec], dtype=np.single)
    f.attrs['M1_PointingRADEC_deg'] = np.array(Pointing_RaDec[0], dtype=np.single)
    f.attrs['M2_PointingRADEC_deg'] = np.array(Pointing_RaDec[1], dtype=np.single)
    
    dsets = []
    dsets.append(f.create_dataset("dl1/event_id", data=eventstream_corrected))#, compression_opts=9)
    
    dt = np.dtype([("M1", np.bool), ("M2", np.bool)])
    dsets.append(f.create_dataset("trig/tels_with_trigger", (nevents_total,), dt))
    dsets[-1][...] = np.array(tels_with_data, dtype = dt)
    f["/dl1/tels_with_data"] = h5py.SoftLink('/trig/tels_with_trigger')
    
    dt = np.dtype([("M1_AzCorr", np.single), ("M1_AltCorr", np.single), 
                   ("M2_AzCorr", np.single), ("M2_AltCorr", np.single)])
    dsets.append(f.create_dataset("pointing", (nevents_total,), dt))
    dsets[-1][...] = np.array(pointing, dtype = dt)
    dsets[-1].attrs['FIELD_0_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_1_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_2_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_3_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_4_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_5_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_6_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_7_UNIT'] = np.string_("deg")
    
    dt = np.dtype([("M1_RaCorr", np.single), ("M1_DecCorr", np.single), ("M2_RaCorr", np.single), ("M2_DecCorr", np.single)])
    dsets.append(f.create_dataset("pointing_radec", (nevents_total,), dt))
    dsets[-1][...] = np.array(pointing_radec, dtype = dt)
    dsets[-1].attrs['FIELD_0_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_1_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_2_UNIT'] = np.string_("deg")
    dsets[-1].attrs['FIELD_3_UNIT'] = np.string_("deg")
    
    
    
    
    dt = np.dtype([("M1_mjd", np.int32), ("M1_sec", np.double), ("M2_mjd", np.int32), ("M2_sec", np.double)])
    dsets.append(f.create_dataset("trig/gps_time", (nevents_total,), dt))
    dsets[-1][...] = np.array(gps_time, dtype = dt)
    dsets[-1].attrs['FIELD_0_UNIT'] = np.string_("d")
    dsets[-1].attrs['FIELD_1_UNIT'] = np.string_("s")
    dsets[-1].attrs['FIELD_2_UNIT'] = np.string_("d")
    dsets[-1].attrs['FIELD_3_UNIT'] = np.string_("s")
    
    dt = np.dtype([("M1", np.double), ("M2", np.double)])
    dsets.append(f.create_dataset("inst/subarray/tel_coords", (3,), dt))
    dsets[-1][...] = np.array(TelPos_2D, dtype = dt)
    dsets[-1].attrs['FIELD_0_UNIT'] = np.string_("m")
    dsets[-1].attrs['FIELD_1_UNIT'] = np.string_("m")
    dsets[-1].attrs['FIELD_0_NAME'] = np.string_("coordinates (x,y,z)")
    dsets[-1].attrs['FIELD_1_NAME'] = np.string_("coordinates (x,y,z)")
    f["inst/subarray"].attrs['OpticsDescription'] = np.string_("MAGIC")
    f["inst/subarray"].attrs['CameraGeometry'] = np.string_("MAGICCam")
    
    dsets.append(f.create_dataset("weather/air_temperature", data=np.array(air_temperature, dtype=np.single)))#, compression_opts=9)
    dsets[-1].attrs['FIELD_0_UNIT'] = np.string_("deg_C")
    dsets.append(f.create_dataset("weather/air_pressure", data=np.array(air_pressure, dtype=np.single)))#, compression_opts=9)
    dsets[-1].attrs['FIELD_0_UNIT'] = np.string_("hPa")
    dsets.append(f.create_dataset("weather/air_humidity", data=np.array(air_humidity, dtype=np.single)))#, compression_opts=9)
    dsets[-1].attrs['FIELD_0_UNIT'] = np.string_("percent")
    
    for ntel in range(ntels):
        
        ##########################
        # write data to hdf5 file:
        
    
        
        dsets.append(f.create_dataset("dl1/tel"+ str(ntel+1) +"/image", data=arr_phe_3D_data[:,ntel,:], compression='gzip'))#, compression_opts=9)
        dsets.append(f.create_dataset("dl1/tel"+ str(ntel+1) +"/peakpos", data=arr_arrival_3D_data[:,ntel,:], compression='gzip'))#, compression_opts=9)
        dsets.append(f.create_dataset("dl1/tel"+ str(ntel+1) +"/badpixels", data=arr_badpixels_3D_data[ntel].transpose(), compression='gzip'))#, compression_opts=9)
        dsets[-1].attrs['FIELD_00_NAME'] = np.string_("UnsuitableRun")
        dsets[-1].attrs['FIELD_01_NAME'] = np.string_("UnsuitableEvt")
        dsets[-1].attrs['FIELD_02_NAME'] = np.string_("UnreliableRun")
        dsets[-1].attrs['FIELD_03_NAME'] = np.string_("HiGainBad")
        dsets[-1].attrs['FIELD_04_NAME'] = np.string_("LoGainBad")
        dsets[-1].attrs['FIELD_05_NAME'] = np.string_("UnsuitableCalLevel")
        dsets[-1].attrs['FIELD_06_NAME'] = np.string_("UnreliableCalLevel")
        dsets[-1].attrs['FIELD_07_NAME'] = np.string_("Uncalibrated:HiGainNotFitted")
        dsets[-1].attrs['FIELD_08_NAME'] = np.string_("Uncalibrated:HiLoGainNotFitted")
        dsets[-1].attrs['FIELD_09_NAME'] = np.string_("Uncalibrated:HiGainOscillating")
        dsets[-1].attrs['FIELD_10_NAME'] = np.string_("Uncalibrated:LoGainOscillating")
        dsets[-1].attrs['FIELD_11_NAME'] = np.string_("Uncalibrated:LoGainSaturation")
        dsets[-1].attrs['FIELD_12_NAME'] = np.string_("Uncalibrated:ChargeIsPedestal")
        dsets[-1].attrs['FIELD_13_NAME'] = np.string_("Uncalibrated:ChargeErrNotValid")
        dsets[-1].attrs['FIELD_14_NAME'] = np.string_("Uncalibrated:ChargeRelErrNotValid")
        dsets[-1].attrs['FIELD_15_NAME'] = np.string_("Uncalibrated:ChargeSigmaNotValid")
        dsets[-1].attrs['FIELD_16_NAME'] = np.string_("Uncalibrated:MeanTimeInFirstBin")
        dsets[-1].attrs['FIELD_17_NAME'] = np.string_("Uncalibrated:MeanTimeInLast2Bins")
        dsets[-1].attrs['FIELD_18_NAME'] = np.string_("Uncalibrated:DeviatingNumPhes")
        dsets[-1].attrs['FIELD_19_NAME'] = np.string_("Uncalibrated:RelTimeNotFitted")
        dsets[-1].attrs['FIELD_20_NAME'] = np.string_("Uncalibrated:RelTimeOscillating")
        dsets[-1].attrs['FIELD_21_NAME'] = np.string_("Uncalibrated:DeviatingNumPhots")
        dsets[-1].attrs['FIELD_22_NAME'] = np.string_("Uncalibrated:HiGainOverFlow")
        dsets[-1].attrs['FIELD_23_NAME'] = np.string_("Uncalibrated:LoGainOverFlow")
        dsets[-1].attrs['FIELD_24_NAME'] = np.string_("Uncalibrated:UnsuitableDC")
        dsets[-1].attrs['FIELD_25_NAME'] = np.string_("Unsuitable")
        
        # Raw-level data with non-data events for cross checks:
        dt = np.dtype([(TrigPatternNames[0], np.bool), (TrigPatternNames[1], np.bool), (TrigPatternNames[2], np.bool), 
                       (TrigPatternNames[3], np.bool), (TrigPatternNames[4], np.bool), (TrigPatternNames[5], np.bool), 
                       (TrigPatternNames[6], np.bool), (TrigPatternNames[7], np.bool)])
        dsets.append(f.create_dataset("r0/tel"+ str(ntel+1) +"/TrigPattern", (nevents[ntel],), dt))
        dsets[-1][...] = np.array(TrigPattern_2D[ntel], dtype = dt)
        dsets.append(f.create_dataset("r0/tel"+ str(ntel+1) +"/DAQEvtNumber", data=DAQEvtNumber_2D[ntel]))
        dsets.append(f.create_dataset("r0/tel"+ str(ntel+1) +"/StereoEvtNumber", data=StereoEvtNumber_2D[ntel]))
        dsets.append(f.create_dataset("r0/tel"+ str(ntel+1) +"/ClockCounter", data=ClockCounter_2D[ntel]))
        dsets.append(f.create_dataset("r0/tel"+ str(ntel+1) +"/TimeDiff", data=TimeDiff_2D[ntel]))
        
    # write MC information:
    if is_mc == True and ntel == 1:
        dsets_mcheader = []
    
        dt = np.dtype([("Energy", np.double),])
        dsets_mcheader.append(f.create_dataset("mc/energy", (nevents_total,), dt))
        dsets_mcheader[-1][...] = np.array(energy, dtype = dt)
        dsets_mcheader[-1].attrs['FIELD_0_NAME'] = np.string_("Energy")
        dsets_mcheader[-1].attrs['FIELD_0_UNIT'] = np.string_("TeV")
    
        dt = np.dtype([("Core_x", np.double), ("Core_y", np.double)])
        dsets_mcheader.append(f.create_dataset("mc/core_xy", (nevents_total,), dt))
        dsets_mcheader[-1][...] = np.array(core_xy, dtype = dt)
        dsets_mcheader[-1].attrs['FIELD_0_NAME'] = np.string_("Core_x")
        dsets_mcheader[-1].attrs['FIELD_1_NAME'] = np.string_("Core_y")
        dsets_mcheader[-1].attrs['FIELD_0_UNIT'] = np.string_("m")
        dsets_mcheader[-1].attrs['FIELD_1_UNIT'] = np.string_("m")
    
        dt = np.dtype([("H_first_int", np.double),])
        dsets_mcheader.append(f.create_dataset("mc/h_first_int", (nevents_total,), dt))
        dsets_mcheader[-1][...] = np.array(h_first_int, dtype = dt)
        dsets_mcheader[-1].attrs['FIELD_0_NAME'] = np.string_("H_first_int")
        dsets_mcheader[-1].attrs['FIELD_0_UNIT'] = np.string_("m")
            
    
    f.close()




if __name__ == '__main__':
    
    main(sys.argv[1:])

##    end of file    ##########################################################
############################################################################### 