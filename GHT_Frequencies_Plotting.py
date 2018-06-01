# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:37:57 2017
Used to define the frequencies of glaciohydraulic tremor as well as the wave type

@author: vore5101
"""
#%% packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import datetime as dt
from obspy.core.utcdatetime import UTCDateTime
from UTCDateTime_funcs import UTC2dn
from matplotlib import dates as mdates

#----------------------------------------------------------------------------
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result
    
#%% User Defined Variables
Station='ETIP'
year=2016
day_range=np.arange(130,270,1)
Freqmin=1.5
Freqmax=10

#%% Plot Constraints
title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'bold',
                  'verticalalignment':'bottom'}
axis_font = {'fontname':'Arial', 'size':'12','weight':'bold'}

fig,ax=plt.subplots(figsize=(8,6))

#%% Used if you want to plot the PSD with constraints 

#with open('mpZQ_RTBD.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
#    t, t_dt64,freqs, Pdb_array, pp, data_dir, station = pickle.load(f)
#
##%% Plot the output of the big runs as median spectrograms
#
#mask_val1 = Pdb_array<=-300
#mask_val2 = np.isinf(Pdb_array)
#
#
#Pdb_array_mask = np.ma.masked_where(np.any([mask_val1, mask_val2], axis=0), Pdb_array)
#
#t_datenum = UTC2dn(t) # Convert a np.array of obspy UTCDateTimes into datenums for the purpose of plotting
#
##plt.imshow(np.log10(Pxx_vals[0:-2,]), extent = [0, len(file_names), freqs[1], freqs[freq_nums-1]])
#fig, ax = plt.subplots()#figsize=(8, 4))
#qm = ax.pcolormesh(t_datenum, freqs, Pdb_array_mask, cmap='YlOrRd')#, extent = [0, len(file_names), freqs[1], freqs[freq_nums-1]])
#ax.set_ylim([1,10])
#ax.set_ylabel('Frequency (Hz)')
#
## Set the date limits for the plot, to make each station's plots consistent
#ax.set_xlim(mdates.date2num([dt.date(2016, 5, 24), dt.date(2016, 7, 18)]))
#
## Format the xaxis of the pcolormesh to be dates
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#fig.autofmt_xdate()
#
#qm.set_clim(vmin=-190, vmax=-140)
#cb = plt.colorbar(qm, ticks=np.arange(-190,-140, 10))
#cb.set_label('Power (dB rel. 1 ($m^2$/$s^2$/Hz))')
#plt.title(station)
#
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.autofmt_xdate()

#%% Ploting Constraints
for dayofyear in day_range: 
    

    d=dt.date(year,1,1) + dt.timedelta(dayofyear)
                               
    UTCDay=UTCDateTime(year=year, julday=dayofyear, hour=0, minute=0)


    # Opens all constraints that exceed the threshold value for a given day 
    try: 
        
        #Power
        with open('F:/Research/Constraints/%s/Power/PP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqsPower_cut,Power_average,PeakPower_freq,PeakPower_values = pickle.load(f)
        
        #Singular Values
        with open('F:/Research/Constraints/%s/SV/SVP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqSV, SV, SVPeak_values, SVPeak_freq = pickle.load(f)
        
        #Rayleigh waves
        with open('F:/Research/Constraints/%s/Rayleigh/RP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqR, PhiRayleigh, RPeak_values, RPeak_freq = pickle.load(f)
        
        #Body waves
        with open('F:/Research/Constraints/%s/020/ZP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqZ, Phi020, ZPeak_values, ZPeak_freq = pickle.load(f)     
        
        #Other waves
        with open('F:/Research/Constraints/%s/Other/OP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
                freqO, PhiOther , OPeak_values, OPeak_freq = pickle.load(f)
                
    except IOError:
        continue
#    
   

    
    
#%% Used to make figure 3
    
#    fig, axs = plt.subplots(5, 1, sharex=True,figsize=(10,9))
##   fig.subplots_adjust(hspace=0)
#    
#    axs[0].plot(freqsPower_cut, Power_average, color='k')
#    axs[0].scatter(PeakPower_freq,PeakPower_values,color='k')
#    axs[0].set_ylabel('Power [Db/Hz]',**axis_font)
#    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
#    
#    axs[1].plot(freqSV,SV,color='k')
#    axs[1].scatter(SVPeak_freq,SVPeak_values,color='k')
#    axs[1].set_ylabel('SV0/SV1 [ ]',**axis_font)
#    axs[1].set_ylim(ymin=0)
#    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(1))
#    
#    axs[2].plot(freqR,PhiRayleigh,color='k')
#    axs[2].scatter(RPeak_freq,RPeak_values,color='b')
#    axs[2].set_ylabel('Rayleigh [%]',**axis_font)
#    axs[2].set_ylim(ymin=0)
#    axs[2].yaxis.set_major_locator(ticker.MultipleLocator(20))
#    
#    axs[3].plot(freqZ,Phi020, color='k')
#    axs[3].scatter(ZPeak_freq,ZPeak_values, color='r')
#    axs[3].set_ylabel('P,S,Love [%]',**axis_font)
#    axs[3].set_ylim(ymin=0)
#    axs[3].yaxis.set_major_locator(ticker.MultipleLocator(20))
#    
#    axs[4].plot(freqO,PhiOther,color='k')
#    axs[4].scatter(OPeak_freq,OPeak_values, color='g')
#    axs[4].set_ylabel('Other [%]',**axis_font)
#    axs[4].set_ylim(ymin=0)
#    axs[4].yaxis.set_major_locator(ticker.MultipleLocator(20))
#    axs[4].set_xlabel('Frequency [Hz]',**axis_font)
#    axs[4].set_xlim(xmin=Freqmin,xmax=Freqmax)
#    dayofyear=dayofyear+1
#    

#------------------------------------------------------------------------------
#Used to find the frequencies that contain glaciohydraulic tremor
    
    freq= freqSV    
    try: PeakPower_values[0]
    except IndexError: continue

#%%   This transforms power to match frequencies of other constraints          
    # Power was sampled at a finer frequency scale than the other constraints.
    Index=[]
    for x in PeakPower_values:
        Index.append(np.where(np.array(Power_average)==x)[0][0])
    
    HighPowerGroup=group_consecutives(np.array(Index))
    PowerFreq=[]
    for H in np.arange(0,len(HighPowerGroup)):
        freqsHighPower=np.take(freqsPower_cut, HighPowerGroup[H])
        Loc=np.where(np.logical_and(np.greater_equal(freq,min(freqsHighPower)),np.less_equal(freq,max(freqsHighPower))))[0]
        PowerFreq.append(np.take(np.array(freq),Loc))
    FreqPeak_Power=np.concatenate(PowerFreq).tolist()

#%%  defines the wave type of the tremor signal 
    Freq020wave=[];FreqRayleighwave=[];FreqOtherwave=[]
    
    for V in ZPeak_freq:                                                        #Find frequency of body wave glaciohydraulic tremor
        if V in FreqPeak_Power and V in SVPeak_freq:
            Freq020wave.append(V)
            
    for V in RPeak_freq:                                                        #Find frequency of Rayligh wave glaciohydraulic tremor
        if V in FreqPeak_Power and V in SVPeak_freq:
            FreqRayleighwave.append(V)
            
    for V in OPeak_freq:                                                        # Find frequency of mixed wave glaciohydraulic tremor
        if V in FreqPeak_Power and V in SVPeak_freq:
            FreqOtherwave.append(V)
    
    # Creates array of dayofyear for plotting purposes
    Day_array_020=[UTCDay]*len(Freq020wave)
    Day_array_Rayleigh=[UTCDay]*len(FreqRayleighwave)
    Day_array_other=[UTCDay]*len(FreqOtherwave)   
    
    #Convert day array to datenum format
    Day020_datenum = UTC2dn(Day_array_020)
    DayRayleigh_datenum = UTC2dn(Day_array_Rayleigh)
    DayOther_datenum = UTC2dn(Day_array_other)

#%% Save frequencies of different wave type of glaciohydraulic tremor    

#    if len(Freq020wave)!= 0:
#        with open('E:/Research/Results/%s/freqs_by_WT/020/freq020_%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
#            pickle.dump([Freq020wave], f)

#    if len(FreqRayleighwave)!=0:
#        with open('E:/Research/Results/%s/freqs_by_WT/Rayleigh/freqRayleigh_%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
#            pickle.dump([FreqRayleighwave], f)

#    if len(FreqOtherwave)!= 0:
#        with open('E:/Research/Results/%s/freqs_by_WT/Other/freqOther_%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
#            pickle.dump([FreqOtherwave], f)
        
    
#%% Create a scatter plot of frequencies of glaciohydraulic tremor
    
    if dayofyear != max(day_range):
        ax.scatter(Day020_datenum,Freq020wave,color='r',s=15)        
        ax.scatter(DayOther_datenum,FreqOtherwave,color='g',s=15,)
        ax.scatter(DayRayleigh_datenum,FreqRayleighwave,color='b',s=15)
    else:
        ax.scatter(Day020_datenum,Freq020wave,color='r',label='P,S,Love',s=15)        
        ax.scatter(DayOther_datenum,FreqOtherwave,color='g',label='Other',s=15,)
        ax.scatter(DayRayleigh_datenum,FreqRayleighwave,color='b',label='Rayleigh',s=15)
    
    
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.autofmt_xdate()






















