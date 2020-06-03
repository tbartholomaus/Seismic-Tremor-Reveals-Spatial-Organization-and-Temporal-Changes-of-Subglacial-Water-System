# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:04:33 2017

@author: vore5101

Passes a three component seismic signal through a fourier transform with a multitaper
To be used in tandem with the PolarPlot_GHT_freqs.py. This method is adapted from Park et. al [1983] 
"""

#%% Packages needed

from pathlib import Path

import numpy as np
from numpy.fft import fft
from numpy.fft import fftfreq 
from obspy import read
from obspy.core import UTCDateTime
from spectrum import*
import pickle
import datetime
import configparser


#%% READ FROM THE PAR FILE
config = configparser.ConfigParser()
config.read('polarization.par')

Station = config['DEFAULT']['station']
Network = config['DEFAULT']['network']

year = int(config['DEFAULT']['year'])
start_month = int(config['DEFAULT']['start_month'])
end_month = int(config['DEFAULT']['end_month'])
start_day = int(config['DEFAULT']['start_day'])
end_day = int(config['DEFAULT']['end_day'])

averaging_number = int(config['DEFAULT']['averaging_number']) # number of spectral covariance matricies that are averaged together

Freqmin = float(config['DEFAULT']['Freqmin']) # Minimum frequancy of interest {Hz}
Freqmax = float(config['DEFAULT']['Freqmax']) # Max frequency of interest [Hz]


#%% User defined Variables
# data_path = Path('/Volumes/labdata/basic_data/seismic_data/day_vols/LEMON/')
data_path = Path('/data/stor/basic_data/seismic_data/day_vols/LEMON/')
analysis_path = Path.cwd()
                                                               
WLength=60                                                                      # Window length of sampling [sec]
step=30                                                                         # Overlap of winows [sec]
RunTime= 60*60*24                                                               # run time for a given dat [sec]

start_day = datetime.datetime(year, start_month, start_day, 0,0,0)
end_day = datetime.datetime(year, end_month, end_day, 0,0,0)

dates = np.arange(start_day, end_day, datetime.timedelta(days=1))


# %%
Path.exists(analysis_path / 'FFT_results' / Station)

if not Path.exists(analysis_path / 'FFT_results' / Station):
    Path.mkdir(analysis_path / 'FFT_results' / Station)

# %% Splits data set into defined windows
#Vert=read('/mnt/lfs2/tbartholomaus/Seis_data/day_vols/TAKU/SV03/'+Station+'/'+Station+'.ZQ..HHZ.%s.%03d'%(year,day))
#North=read('/mnt/lfs2/tbartholomaus/Seis_data/day_vols/TAKU/SV03/'+Station+'/'+Station+'.ZQ..HHN.%s.%03d'%(year,day))
#East=read('/mnt/lfs2/tbartholomaus/Seis_data/day_vols/TAKU/SV03/'+Station+'/'+Station+'.ZQ..HHE.%s.%03d'%(year,day)) 

full_path = data_path / Station
for day in dates:

    # dt=datetime.datetime(year,month,day, hr_s,min_s,sec_s)                      # puts date into datetime format
    temp = (day - np.datetime64(str(year-1) + '-12-31')).astype('timedelta64[D]')                                            # defines the day of year
    doy = int( temp / np.timedelta64(1, 'D') )
    try:                                                                        # reads in three channels of signal if the file exists
                                                                                # highpass filer (user change change filter limit)
        Vert= read(str(full_path) + '/%s.%s..HHZ.%s.%03d'%(Station, Network, year, doy)).filter('highpass', freq=0.5)
        North= read(str(full_path) + '/%s.%s..HHN.%s.%03d'%(Station, Network, year, doy)).filter('highpass', freq=0.5)
        East= read(str(full_path) + '/%s.%s..HHE.%s.%03d'%(Station, Network, year, doy)).filter('highpass', freq=0.5)
        Vert.merge(fill_value='latest');North.merge(fill_value='latest');East.merge(fill_value='latest')
        sample_rate = Vert[0].stats.sampling_rate  # Hz  Sampling rate of instrumentation
        
        if len(Vert[0])< sample_rate*60*60*22:                                  # double check record is at least 22hrs in length
            continue
        # Trim_s=UTCDateTime("%s-%03dT%02d:%02d:%02d"%(year,doy,hr_s,min_s,sec_s)) 
        # Trim_e=Trim_s+(RunTime)
        Vert.detrend(type='linear'); North.detrend(type='linear'); East.detrend(type='linear') # detrend data set
        # Vert.trim(Trim_s,Trim_e);North.trim(Trim_s,Trim_e);East.trim(Trim_s,Trim_e)           # trim data set
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
    while len(WindowLoop) % averaging_number != 0:                              # cuts windows so they are divisable by the averaging number (user defined) 
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
    with open(str(analysis_path / 'FFT_results') + '/%s/FFT.%s.pickle'%(Station,doy), 'wb') as f:  
        pickle.dump([FFT_stack_real[0:FreqIx,:,:],FFT_stack_imag[0:FreqIx,:,:],WindowLoop,Frequency[0:FreqIx],averaging_number], f)

    
        
        
        
        
        
        

