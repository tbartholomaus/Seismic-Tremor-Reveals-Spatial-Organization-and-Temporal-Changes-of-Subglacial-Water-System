# -*- coding: utf-8 -*-
"""
Created on Tue May 01 12:16:17 2018
This script creates plots a time series of the glaciohydraulic tremor constrains within a given frequency band. 

@author: vore5101
"""
#%% Packages Needed 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime as dt
from obspy.core.utcdatetime import UTCDateTime

#%% User Defined Variables 
Station='RTBD'              # Seismic Station 
year=2016                   # Year of data collection 
#month=7
day_range= np.arange(141,200,1) # Day of year range
Freqmin=2.35                    # Frequency minimum [Hz]
Freqmax= 2.7                      # Frequency Maximum [Hz]

# Title of plot constraints 
title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'bold',
                  'verticalalignment':'bottom'}
                  
# Axis font constraints 
axis_font = {'fontname':'Arial', 'size':'12','weight':'bold'}

#%% Open all the constraints for each day for plotting of time series

# Initiate variable arrays 
day=[]; PowerM=[]; SVM=[]
RayleighM=[]; Phi020M=[]; OtherM=[]

for dayofyear in day_range: 
    
    UTCDay=UTCDateTime(year=year, julday=dayofyear, hour=0, minute=0) #UTC Date Time format

    # Open all the GHT Constraints for the day
    try: 
        # Power
        with open('D:/Research/Constraints/%s/Power/PP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqsPower_cut,Power_average,PeakPower_freq,PeakPower_values = pickle.load(f)
        # Find powers within the given frequency range 
        idxPower=np.where(np.all([np.array(freqsPower_cut)> Freqmin,np.array(freqsPower_cut)<Freqmax], axis=0))[0]
        freqsPower=freqsPower_cut[idxPower];Power_average=Power_average[idxPower]
        # find median power for within the frequency for the given day 
        Power_average_mean=np.median(Power_average)
        
        # Singular values 
        with open('D:/Research/Constraints/%s/SV/SVP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqSV, SV, SVPeak_values, SVPeak_freq = pickle.load(f)
        # Finds singular value ratios within the given frequency range
        idxSV=np.where(np.all([np.array(freqSV)> Freqmin,np.array(freqSV)<Freqmax], axis=0))[0]
        SV=SV[idxSV]
        # find median of the singular value ratios within the frequency for a given day
        SV_mean=np.median(SV)
        
        # Rayleigh waves 
        with open('D:/Research/Constraints/%s/Rayleigh/RP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqR, PhiRayleigh, RPeak_values, RPeak_freq = pickle.load(f)
        # Finds Rayleigh wave percentages within the given frequency range
        idxR=np.where(np.all([np.array(freqR)> Freqmin,np.array(freqR)<Freqmax], axis=0))[0]
        PhiRayleigh=PhiRayleigh[idxR]
        # find median of the Rayleigh wave percentages within the frequency for a given day
        PhiRayleigh_mean=np.median(PhiRayleigh)
        
        #Body Waves
        with open('D:/Research/Constraints/%s/020/ZP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqZ, Phi020, ZPeak_values, ZPeak_freq = pickle.load(f) 
        # Finds Body wave percentages within the given frequency range
        idxZ=np.where(np.all([np.array(freqZ)> Freqmin,np.array(freqZ)<Freqmax], axis=0))[0]
        Phi020=Phi020[idxZ]
        # find median of the body wave percentages within the frequency for a given day
        Phi020_mean=np.median(Phi020)
        
        #mixed waves
        with open('D:/Research/Constraints/%s/Other/OP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqO, PhiOther , OPeak_values, OPeak_freq = pickle.load(f)
        # Finds Mixed wave percentages within the given frequency range
        idxO=np.where(np.all([np.array(freqO)> Freqmin,np.array(freqO)<Freqmax], axis=0))[0]
        PhiOther=PhiOther[idxO]
        # find median of the mixed wave percentages within the frequency for a given day
        PhiOther_mean=np.median(PhiOther)
        
    except IOError:
        continue
#    
    freq= freqSV
    
    # record each constraint for that day
    day.append(dayofyear)
    PowerM.append(Power_average_mean)
    SVM.append(SV_mean)
    RayleighM.append(PhiRayleigh_mean)
    Phi020M.append(Phi020_mean)
    OtherM.append(PhiOther_mean)


#%% Plotting the time series[ note: x axis in day of year]   
fig, axs = plt.subplots(5, 1, sharex=True,figsize=(10,9))

axs[0].plot(day, PowerM, color='k')
axs[0].set_ylabel('Power [Db/Hz]',**axis_font)

axs[1].plot(day,SVM,color='k')
axs[1].set_ylabel('SV0/SV1 [ ]',**axis_font)
axs[1].set_ylim(ymin=0)
axs[1].axhline(y=2.5, color='r')

axs[2].plot(day,RayleighM,color='k')
axs[2].set_ylabel('Rayleigh [%]',**axis_font)
axs[2].set_ylim(ymin=0)
axs[2].axhline(y=39, color='red')

axs[3].plot(day, Phi020M, color='k')
axs[3].set_ylabel('Body',**axis_font)
axs[3].set_ylim(ymin=0)
axs[3].axhline(y=39, color='red')

axs[4].plot(day,OtherM,color='k')
axs[4].set_ylabel('Other [%]',**axis_font)
axs[4].set_ylim(ymin=0)
axs[4].axhline(y=96,color='red')
#  


    
    
    
    
    
    
    
    
    
