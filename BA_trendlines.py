# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:17:44 2017
@author: Margot E Vore

This code is used for finding tendlines of temporal changes in backazimuth 
measurements. The median backazimuthis found for a given band of Glaciohydraulic
tremor on a given day for the all days of data. A trendline is then placed 
through the data and R^2 is calculated.

This code was not used to make any figures in the paper but was used for inital analysis of data set 


"""
#%% Packages Needed 

import pickle
import numpy as np
import matplotlib.pyplot as plt
from obspy.core.utcdatetime import UTCDateTime
from UTCDateTime_funcs import UTC2dn
from matplotlib import dates as mdates
from scipy.stats import circmean
import os

# %% Trendline function
def trendline(xd, yd, N, order=1, c='r', alpha=1, Rval=False):
    """Make a line of best fit and calculates R^2 for the trendline"""

    coeffs = np.polyfit(xd, yd, order)                                          #Calculate trendline


    intercept  = coeffs[-1]
    slope      = coeffs[-2]
    power      = coeffs[0] if order == 2 else 0


    minxd = np.min(xd)
    maxxd = np.max(xd)


    xl = np.array([minxd, maxxd])
    yl = power * xl ** 2 + slope * xl + intercept


    axs[N].plot(xl, yl, c, alpha=alpha,color='k')                               #Plot trendline


    p = np.poly1d(coeffs)


    ybar  = np.sum(yd) / len(yd)                                                #Calculate R Squared
    ssreg = np.sum((p(xd) - ybar) ** 2)
    sstot = np.sum((yd - ybar) ** 2)
    Rsqr  = ssreg / sstot


    if not Rval:
        #Plot R^2 value
        axs[N].text(175,134, '$R^2 = %0.2f$' % Rsqr)                            # USER: set y and x limits of text
        axs[N].text(175,136,'$y=%0.3fx + %0.3f$'%(slope,intercept))             # USER: set y and x limits of text
        return Rsqr
    else:
        return Rsqr                                                             #Return the R^2 value:
        
# %% Define Variables

Station    = 'RTBD'                                                             # Station Name
year       = 2016                                                               # year of data collection
Freqmin    = [2.35,3.2]                                                         # Frequency bands of GHT min
Freqmax    = [2.7,3.7]                                                          # Frequency bands of GHT max
day_range  = np.arange(172,275,1)                                               # Day Range

#%%

fig,axs = plt.subplots(len(Freqmin),1,sharex=True)
R2=[]


for N in np.arange(0,len(Freqmin)):                                             # loop through all Frequency Bands


    Min_Bin_avg = []; Max_Bin_avg = [];                                         # Initiate variables
    day = []; Median_Bin_avg = []
    
    for dayofyear in day_range:                                                 # Loops through day of year
    
    
        UTCDay = UTCDateTime(year=year, julday=dayofyear, hour=0, minute=0)    # Convert DOY to DateTime format
    
#%% Opening data for stations with Rayleigh wave and other waveform bands
        if Station == 'GIW3':                                                  # USER: Change Path Names
            A = 'E:/Research/Results/%s/freqs_by_WT/Rayleigh/freqRayleigh_%s.pickle'\
                %(Station,dayofyear)                                         
            B = 'E:/Research/Results/%s/freqs_by_WT/Other/freqOther_%s.pickle'\
                %(Station,dayofyear)
                
        
            if os.path.isfile(A) == False and os.path.isfile(B)== False:        # Determines if day has Rayleigh wave or other wave GHT
                continue

            
            if os.path.isfile(A) == True:                                       # Open frequencies of Rayleigh GHT
                with open(A,'rb') as f:
                    GHT_freq_Rayleigh = pickle.load(f)
                GHT_freq_Rayleigh = GHT_freq_Rayleigh[0]
            
                    
            if os.path.isfile(B) == True:                                       # Open frequencies of Other GHT
                with open(B,'rb') as f:
                    GHT_freq_Other = pickle.load(f)
                GHT_freq_Other = GHT_freq_Other[0]

                
            with open('E:/Research/Results/%s/Backazimuths/BA_Combo.%s.pickle'\
                %(Station,dayofyear), 'rb') as f:                               # Open Backazimuth measurements  
                Range,MaxPD,PD_tremor = pickle.load(f)                          # freq first for BBWL,GIW3   
            
            
            if os.path.isfile(A) == True and os.path.isfile(B) == True:         # combine Other and Rayleigh wave frequencies
                GHT_freq_Rayleigh = np.concatenate((GHT_freq_Rayleigh,GHT_freq_Other))
                
#%% Opening data for stations with ONLY Rayleigh wave bands
                
        else:  
            try:                                                                 # USER: Change Path Names
                with open('E:/Research/Results/%s/Backazimuths/BA.%s.pickle'\
                    %(Station,dayofyear), 'rb') as f:                           # Open Backazimuth Measurements
                    Range,MaxPD,PD_tremor = pickle.load(f) #freq first for BBWL,GIW3
                with open('E:/Research/Results/%s/freqs_by_WT/Rayleigh/freqRayleigh_%s.pickle'\
                    %(Station,dayofyear),'rb') as f:                            # Open frequencies of Rayleigh wave GHT
                    GHT_freq_Rayleigh = pickle.load(f)
            except IOError:
                continue
            
            
            GHT_freq_Rayleigh = GHT_freq_Rayleigh[0]
            
#%%Finding all frequncies between the min and max value of band   
            
        if len(GHT_freq_Rayleigh)>1:                                            # Creating frequency values to iterate over 
            diff = GHT_freq_Rayleigh[1]-GHT_freq_Rayleigh[0]                    # Frequency Step of tremor signal
            Freq = np.arange(min(GHT_freq_Rayleigh),max(GHT_freq_Rayleigh)+diff,diff) 
        else:
            Freq = GHT_freq_Rayleigh
            
         
        Index = np.where(np.logical_and(np.less_equal(Freq,Freqmax[N]),\
            np.greater_equal( Freq, Freqmin[N])))[0]                            # The Index values of frequencies associated with MaxPD 

#%% Record BA Bins that have highest, 2nd highest and 3rd highest probabilities      
        
        try:                                                                    #Occurs only if BA exist within a given frequency band
            MaxPD     = MaxPD[min(Index):max(Index)+1]
            minBin    = np.amin(MaxPD,axis=1)                                  # Finds the 3rd highest BA bin for a frequency
            maxBin    = np.amax(MaxPD,axis=1)                                  # Finds the maximum BA bin for a frequency
            medianBin = np.median(MaxPD,axis=1)                                # Finds the 2nd highest BA bin for a frequency
       except ValueError:
            continue
        
        
        if np.isnan(circmean(np.delete(medianBin,np.where(np.isnan(medianBin))[0]),360,0)):
            continue                                                            # skips days that have nan values as average
        
        
        Min_Bin_avg.append(circmean(np.delete(minBin,np.where\
            (np.isnan(minBin))[0]),360,0))                                      # Finds the average of 3rd highest bin for a day
        Max_Bin_avg.append(circmean(np.delete(maxBin,np.where\
            (np.isnan(maxBin))[0]),360,0))                                      # Finds the average of the maximum bin for a day
        Median_Bin_avg.append(circmean(np.delete(medianBin,np.where\
            (np.isnan(medianBin))[0]),360,0))                                   # Finds the average of 2nd highest bin for a day
        
        
        day.append(dayofyear)                                                   # records day of year
    
#%% Plotting Average Median Bin 
    
    axs[N].scatter(day,Median_Bin_avg,marker='o',color='r',\
        label='%s -%s'%(Freqmin[N],Freqmax[N]))                                # plots median backazimuth bin for a given day and frequency range
#    axs[N].legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Frequency Bands")
#    axs[N].set_ylim(120,150)                                                  #User: Set y-axis Limit
    axs[N].set_ylabel('Backazimuth')
    
    
    R=trendline(day, Median_Bin_avg, N,order=1)                                #Draws trendline of data 
    R2.append(R)

    
axs[N].set_xlabel('Day of year: %s'%year)
axs[N].set_xlim(xmin=170)                                                      #USER: Set x-axis limits
fig.suptitle(Station,fontsize=16)       
 
 
#plt.savefig('C:/Users/vore5101/Desktop/Research/Results/Temporal_Trendlines/%s.png'%Station)
#plt.close()


#%% Frequency Bands of GHT for each Station on Taku and Lemon Creek Glaciers

#BBWU
Freqmin=[2.35,3.0,4.15,4.4]
Freqmax=[2.8,3.35,4.4,4.75]

#BBEU
Freqmin=[1.65]
Freqmax=[2.0]

#BBEL
Freqmin=[2.15,3.25,3.75,4.55,5.2,7.65,9.35]
Freqmax=[2.75,3.7,4.3,5.2,5.8,8.25,9.95]

#BBWL
Freqmin=[1.55,2.0,3.1,5.05,6.4,8.1,9.15]
Freqmax=[1.95,3.05,3.6,5.85,6.9,8.75,9.65]

#ETIP
Freqmin=[3.3,4.0,5.2]
Freqmax=[4.0,4.5,5.6]

#RTBD
Freqmin=[2.35,3.2]
Freqmax=[2.7,3.7]

#TWLV
Freqmin=[1.75,6.55]
Freqmax=[2.05,6.8]

#GIW2
Freqmin=[2.75,8.25]
Freqmax=[3.05,8.75]

#GIW3
Freqmin=[1.75,2.9,6.45,7.25,8.95]
Freqmax=[2.25,3.3,7.1,8.2,9.75]

#GIW4
Freqmin= [1.75,2.0,8.45]
Freqmax=[1.95,2.25,8.7]

#GIW5
Freqmin=[2.15,2.65,6.45]
Freqmax=[2.35,2.80,6.70]
