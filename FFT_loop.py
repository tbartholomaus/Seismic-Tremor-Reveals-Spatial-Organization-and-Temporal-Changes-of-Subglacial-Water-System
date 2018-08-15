# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:04:33 2017

@author: vore5101

Passes a three component seismic signal through a fourier transform with a multitaper
To be used in tandem with the PolarPlot_GHT_freqs.py. This method is adapted from Park et. al [1983] 
"""

#%% Packages needed
import numpy as np
from numpy.fft import fft
from numpy.fft import fftfreq 
from obspy import read
from obspy.core import UTCDateTime
from spectrum import*
import pickle
import datetime

#%% User defined Variables
Station='GIW3'                                                                  # Seiemic Station Name
year=2016                                                                       # Year data was collected
month=7                                                                         # Month of data set
day_range=np.arange(1,32,1)                                                     # Day range of data set
hr_s=00                                                                         # Starting hour of data set
min_s=00                                                                        # starting minute of data set
sec_s= 00                                                                       # starting second of data set
sample_rate=200.0                                                               # Sampling rate of instrumentation
WLength=60                                                                      # Window length of sampling [sec]
step=30                                                                         # Overlap of winows [sec]
RunTime= 60*60*24                                                               # run time for a given dat [sec]
averaging_number=13                                                             # number of matricies to average
                                     
# %% Splits data set into defined windows
#Vert=read('/mnt/lfs2/tbartholomaus/Seis_data/day_vols/TAKU/SV03/'+Station+'/'+Station+'.ZQ..HHZ.%s.%03d'%(year,day))
#North=read('/mnt/lfs2/tbartholomaus/Seis_data/day_vols/TAKU/SV03/'+Station+'/'+Station+'.ZQ..HHN.%s.%03d'%(year,day))
#East=read('/mnt/lfs2/tbartholomaus/Seis_data/day_vols/TAKU/SV03/'+Station+'/'+Station+'.ZQ..HHE.%s.%03d'%(year,day)) 

for day in day_range:

    dt=datetime.datetime(year,month,day, hr_s,min_s,sec_s)                      # puts date into datetime format
    dayofyear=int(dt.strftime('%j'))                                            # defines the day of year
    
    try:                                                                        # reads in three channels of signal if the file exists
                                                                                # highpass filer (user change change filter limit)
        Vert= read('%s.ZQ..HHZ.%s.%03d'%(Station,year, dayofyear)).filter('highpass', freq=0.5)
        North= read('%s.ZQ..HHN.%s.%03d'%(Station,year,dayofyear)).filter('highpass', freq=0.5)
        East= read('%s.ZQ..HHE.%s.%03d'%(Station,year,dayofyear)).filter('highpass', freq=0.5)
        Vert.merge(fill_value='latest');North.merge(fill_value='latest');East.merge(fill_value='latest')
        
        if len(Vert[0])< sample_rate*60*60*22:                                  # double check record is at least 22hrs in length
            continue
        Trim_s=UTCDateTime("%s-%03dT%02d:%02d:%02d"%(year,dayofyear,hr_s,min_s,sec_s)) 
        Trim_e=Trim_s+(RunTime)
        Vert.detrend(type='demean');North.detrend(type='demean'); East.detrend(type='demean') # detrend data set
        Vert.trim(Trim_s,Trim_e);North.trim(Trim_s,Trim_e);East.trim(Trim_s,Trim_e)           # trim data set
    except IOError:
        continue
    
    Vwin=[];Nwin=[];Ewin=[]
    for windowed_Vert in Vert.slide(window_length=WLength,step=step):           # windows vertical component per user defind variables
        windowed_Vert=windowed_Vert[0]
        Vwin.append(windowed_Vert) 
    
    for windowed_North in North.slide(window_length=WLength,step=step):         # windows north component per user defind variables
        windowed_North=windowed_North[0]
        Nwin.append(windowed_North) 
        
    for windowed_East in East.slide(window_length=WLength,step=step):           # windows east component per user defind variables
        windowed_East=windowed_East[0]
        Ewin.append(windowed_East) 
        
        
#%% Perfoms Fourier Transform of each window in each componenet
    StartWin=0
    EndWin= len(Ewin)
    
    WindowLoop= np.arange(StartWin,EndWin,1) 
    H=0
    while len(WindowLoop) % averaging_number != 0:                              # cuts widnows so they are divisiable by the averaging number (user defined) 
        WindowLoop=WindowLoop[:-1]
        H=H+1
        
    # Creating the prolate taper
    NW=2.5                                                                      # user defined   time half bandwidth parameter                                                                    
    [tapers,eigen]=dpss(len(Vwin[0]),NW)                                        # creates eigntapers
    dontuse=np.where (eigen<0.90)                                               # remove tapers that have eigenvalue less than .90
    eigen=np.delete(eigen,dontuse[0])
    tapers=np.delete(tapers,dontuse[0],axis=1)
    n=len(Vwin[1])
    
    # initalizing arrays
    FFT_Z_real=np.empty((len(Vwin[0]),len(eigen)))
    FFT_N_real=np.empty((len(Vwin[0]),len(eigen)))
    FFT_E_real=np.empty((len(Vwin[0]),len(eigen)))
    FFT_Z_imag=np.empty((len(Vwin[0]),len(eigen)))
    FFT_N_imag=np.empty((len(Vwin[0]),len(eigen)))
    FFT_E_imag=np.empty((len(Vwin[0]),len(eigen)))
    FFT_stack_real=np.empty((n,3,len(WindowLoop)))
    FFT_stack_imag=np.empty((n,3,len(WindowLoop)))
    
    
    try:
        for i in WindowLoop[:-1]:
            Frequency=fftfreq(n,1.0/sample_rate)
            for x in np.arange(0,len(eigen),1):
                FFT_Z=fft(tapers[:,x]*Vwin[i])/len(Vwin[i])                     # perform fourier transform with all multi tapers
                FFT_N=fft(tapers[:,x]*Nwin[i])/len(Nwin[i])
                FFT_E=fft(tapers[:,x]*Ewin[i])/len(Ewin[i])
                FFT_Z_real[:,x]=FFT_Z.real
                FFT_N_real[:,x]=FFT_N.real
                FFT_E_real[:,x]=FFT_E.real
                FFT_Z_imag[:,x]=FFT_Z.imag
                FFT_N_imag[:,x]=FFT_N.imag
                FFT_E_imag[:,x]=FFT_E.imag
            FFT=np.mean(FFT_Z_real,1)+1j*np.mean(FFT_Z_imag,1)                  # find the average of the multitaper results
            FFT=np.c_[FFT,np.mean(FFT_N_real,1)+1j*np.mean(FFT_N_imag,1)]
            FFT=np.c_[FFT,np.mean(FFT_E_real,1)+1j*np.mean(FFT_E_imag,1)]
            FFT_stack_real[:,:,i]=np.asmatrix(FFT.real)                         # save real part of fourier transform
            FFT_stack_imag[:,:,i]=np.asmatrix(FFT.imag)                         # save imaginary part of fouier transform
    except ValueError:
        continue
    FreqIx=np.where(np.array(Frequency)>=20)[0][0]                              # used to save data less than 20Hz to save space                           
   
    # save fft results
    with open('E:/FFT_results/%s/FFT.%s.pickle'%(Station,dayofyear), 'wb') as f:  
        pickle.dump([FFT_stack_real[0:FreqIx,:,:],FFT_stack_imag[0:FreqIx,:,:],WindowLoop,Frequency[0:FreqIx],averaging_number], f)

    
        
        
        
        
        
        

