#!/bin/env python
import numpy

from datetime import datetime
import bisect

# Data and documentation
# http://www.esrl.noaa.gov/raobs/intl/Data_request.cgi?byr=2009&bmo=1&bdy=1&bhr=0&eyr=2014&emo=10&edy=6&ehr=14&shour=All+Times&ltype=All+Levels&wunits=Tenths+of+Meters%2FSecond&access=WMO+Station+Identifier
# http://www.esrl.noaa.gov/raobs/intl/fsl_format-new.cgi

####################################
## @brief Class to store a Radio sonde sounding
#
class pSounding(object):
    ####################################
    ## @brief Constructor for a radio sonde sounding
    #
    ## @param self
    #  the object instance
    def __init__(self, sounding):
        self.NPoints=0
        self.DateTime=None
        self.WMONumber=None
        self.Pressure=None
        self.Height=None
        self.Temperature=None
        self.Dewpt=None
        self.WindDir=None
        self.WindSpd=None        
        self.__ingest(sounding)
    
    # Compare using DateTime object
    def __comp__(self, other):
        return self.DateTime<other.Datetime
        
    def getKey(self):
        key='%s-%02d-%02d'%(self.DateTime.year,self.DateTime.month,
                self.DateTime.day)
        return key

    def __initArrays(self):
        self.Pressure=numpy.ndarray((self.NPoints),'d')
        self.Height=numpy.ndarray((self.NPoints),'d')
        self.Temperature=numpy.ndarray((self.NPoints),'d')
        self.Dewpt=numpy.ndarray((self.NPoints),'d')
        self.WindDir=numpy.ndarray((self.NPoints),'d')
        self.WdinSpd=numpy.ndarray((self.NPoints),'d')

    
    def __ingest(self, sounding):
         self.DateTime=datetime.strptime(sounding[0].strip(),'254     %H     %d      %b    %Y')
         self.WMONumber=int(sounding[1].split()[2])
         #print(self.DateTime,' ',self.WMONumber)
         # Get arrays length and init
         i=0
         for l in sounding[4:]:             
             if len(l.split())>2:                
                 id=int(l.split()[0].strip())
                 if id==5:
                     P=float(l.split()[1])/10.
                     H=float(l.split()[2])
                     T=float(l.split()[3])/10.+273.
                     if P>0 and H>0 and T>0:
                         i+=1                 
         self.NPoints=i
         self.__initArrays()
         # Parse everything
         i=0
         for l in sounding[4:]:             
             if len(l.split())>2:                
                 id=int(l.split()[0].strip())
                 if id==5:
                     P=float(l.split()[1])/10.
                     H=float(l.split()[2])
                     T=float(l.split()[3])/10.+273.
                     Dwpt=float(l.split()[4])/10.
                     Wdir=float(l.split()[5])
                     Wspd=float(l.split()[6])/10.
                     #print(i,P,H,T,Dwpt,Wdir,Wspd)
                     if P>0 and H>0 and T>0:
                         self.Pressure[i]=P
                         self.Height[i]=H
                         self.Temperature[i]=T
                         self.Dewpt[i]=Dwpt
                         self.WindDir[i]=Wdir
                         self.WdinSpd[i]=Wspd
                         i+=1
                         

    def dump(self, N=10):
        print('-'*60)
        txt=self.DateTime.__str__()
        if N>self.NPoints:
            N=self.NPoints
        for i in range(N):
            txt+='%.1f %.1f %.1f\n'%(self.Height[i], 
                                   self.Pressure[i],
                                   self.Temperature[i])
        print txt
        print('-'*60)
        return txt
            
        
####################################
## @brief Class to manage a LIDAR run
#
class pRadioSondeReader(object):
    ####################################
    ## @brief Constructor for a Lidar run
    #
    ## @param self
    #  the object instance
    def __init__(self):
        self.FileName=None
        self.SoundingDict={}

    ####################################
    ## @brief Read Radiosone data from an ascii file fsl format
    #
    ## @param self
    #  the object instance
    def readFile(self, filename):
        self.FileName=filename
        cont=open(self.FileName,'r').readlines()
        # parse soundings
        soundingsList=[]
        sounding=[]        
        for line in cont:            
            try:
                id=int(line.split()[0])
            except:
                print('Error: ', line)
                id=0
            if id==254:
                if len(sounding)>1:
                    soundingsList.append(sounding)
                sounding=[line]
            else:
                sounding.append(line)
        # ingest soundings
        for one in soundingsList[0:]:
                if len(one)>1:
                    s=pSounding(one)
                    self.SoundingDict[s.DateTime]=s
                    
    ####################################
    ## @brief Get the sounding nearest to the required date
    #
    ## @param self
    #  the object instance
    ## @param datestr
    #  a date in the form year-month-day
    #
    def getSounding(self, datestr):
        date=datetime.strptime(datestr,'%Y-%m-%d')
        keys=self.SoundingDict.keys()
        keys.sort()
        index=bisect.bisect_left(keys, date)
        return self.SoundingDict[keys[index]]

if __name__ == '__main__':
    s=pRadioSondeReader()
    s.readFile("../data/radiosonde_68110_2009_2014.fsl")
    s.getSounding('2009-08-23').dump()
    
