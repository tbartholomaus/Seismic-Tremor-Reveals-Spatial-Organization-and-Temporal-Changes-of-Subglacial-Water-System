# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 07:55:36 2017

This code calculates the backazimuth of the seismic for each frequency that is considered
Rayleigh wave glaciohydraulic tremor based on the work of Park et. al 1987. This code also 
creates rose diagrams of the probability density of backazimuths. 

### SCRIPTS NEEDED TO RUN PRIOR TO RUNNING THIS SCRIPT: GHT_Frequencies_Plotting.py and FFT_loop.py 

@author: vore5101
"""
#%% Packages needed 
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from numpy.linalg import svd
import cmath
import pickle
import numpy.ma as ma
import matplotlib.gridspec as gridspec
import heapq
import datetime as dt
from pathlib import Path
import configparser

#%% function for grouping consecutive lists of numbers 
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
    
#%% User defined Variables
analysis_path = Path.cwd()

#%% READ FROM THE PAR FILE
config = configparser.ConfigParser()
config.read('polarization.par')

Station = config['DEFAULT']['station']
network = config['DEFAULT']['network']

year = int(config['DEFAULT']['year'])
start_month = int(config['DEFAULT']['start_month'])
end_month = int(config['DEFAULT']['end_month'])
start_day = int(config['DEFAULT']['start_day'])
end_day = int(config['DEFAULT']['end_day'])

Freqmin = float(config['DEFAULT']['PolarFreqmin']) # Minimum frequancy of interest {Hz}
Freqmax = float(config['DEFAULT']['PolarFreqmax']) # Max frequency of interest [Hz]
averaging_number = int(config['DEFAULT']['averaging_number']) # number of spectral covariance matricies that are averaged together
bin_size = int(config['DEFAULT']['bin_size']) # size of bins for backazimuth catagorization [ degrees]
MaxNumber = int(config['DEFAULT']['MaxNumber'])

start_day = dt.datetime(year, start_month, start_day, 0,0,0)
end_day = dt.datetime(year, end_month, end_day, 0,0,0)

dates = np.arange(start_day, end_day, dt.timedelta(days=1))

#%% Load Data Needed (found from GHT_Frequencies.py and FFT_loop.py)

Sin0=[];Sin1=[];Sin2=[];SinTot=[]   
 
# loop through for all days of interest within a month 
for day in dates:
    temp = (day - np.datetime64(str(year-1) + '-12-31')).astype('timedelta64[D]')                                            # defines the day of year
    doy = int( temp / np.timedelta64(1, 'D') )    

    # Path name to rayleigh wave frequencies (as found by GHT_Frequencies.py)
    A= str( analysis_path / 'Results' / Station / 'freqs_by_WT'/'Rayleigh' ) + '/freqRayleigh_%s.pickle'%(doy)

    # Path name to mixed wave frequencies (as found by GHT_Frequencies.py)
#    B='E:/Research/Results/%s/freqs_by_WT/Other/freqOther_%s.pickle'%(Station,doy)


    try:
        # LOad rayligh wave frequencies for a given day of year
        with open(A,'rb') as f:
            GHT_freq_Rayleigh= pickle.load(f)
        GHT_freq_Rayleigh=GHT_freq_Rayleigh[0]

        # Load fft results for day of year (as found by FFT_loop.py)
        with open(str( analysis_path / 'FFT_results' / Station ) + '/FFT.%s.pickle'%(doy),'rb') as f:  # Python 3: open(..., 'rb')
            FFT_stack_real,FFT_stack_imag,WindowLoop,Frequency,averaging_number= pickle.load(f)    
    
    # Will skip day of year if no Rayliegh waves occur on that day
    except IOError:
        continue    
    
#%% Initiation of arrays that will be used to store data during loop through frequencies
    
    # make list of FFT frequency indicies that are dominated by Rayleigh waves
    index=[]    
    for B in GHT_freq_Rayleigh:
       index.append(np.where(Frequency==B)[0][0])
      
    # range of indicies for rayligh wave tremor 
    Range=np.arange(min(index),max(index)+1)
    
    # Initiate backazimuth bins
    bins=np.arange(0.0,365.0,bin_size)
    
    #Initate freq, singular value 1,2 and 3
    freq=[];s0=[];s1=[];s2=[];
    
    #Number of calcuated backazimuths that fall within each bin 
    AzBinned=np.empty((len(Range),len(bins)-1))
    
    # Initiate Probability Density of backazimuths
    PD=np.empty((len(Range),len(bins)-1)) 
    
    #Initiate probabiliy desnity of x number of highest pribabilities (as defined by user in Max Number)
    MaxPD=np.empty((len(Range),MaxNumber))  
    
    # Split Window Loop into groups (size of groups defined by user). This allows for averaging
    Split=np.split(WindowLoop, len(WindowLoop)/float(averaging_number))
 
#%% Averaging fft matricies together to reduce the influence of transient events  
    # Initiate Loop count
    LoopNumber=0
    
    # Loop through frequencies for a given day
    for x in Range : 
        
        # Initiate array to collect backazimuth calculations
        Azimuth=[]
        
        # record frequency 
        freq.append(Frequency[x])
        
        # Initiate number of split groups used
        K=0
        
        
        while K <len(Split): 
            
            # choose subset group to work with from Split array
            SplitLoop=Split[K]  
            
            # Initiate real and imagianry array to hold spectral covariance matricies
            stacked_real=np.empty((3,3,len(SplitLoop)))
            stacked_imag=np.empty((3,3,len(SplitLoop)))
 
         
            #for each array in a group
            for i in SplitLoop: 
                
                # Choose the fft to work with 
                FFT_working=np.asmatrix(FFT_stack_real[:,:,i]+1j*FFT_stack_imag[:,:,i])
                
                # Take conjugate of fft for a frequency
                FFT_T=np.conj(FFT_working[x].T)
                
                # calculate spectral covarience matricie
                SpecCov=np.dot(FFT_T,FFT_working[x])
                
                # define the real and imaginary parts of spectral covarience matrice
                stacked_real[:,:,i-SplitLoop[0]]=SpecCov.real
                stacked_imag[:,:,i-SplitLoop[0]]=SpecCov.imag

            # After looping through the group, average the real and imaginary parts together
            StackMean_real=np.mean(stacked_real,2)
            StackMean_imag=np.mean(stacked_imag,2)
            StackMean=StackMean_real+1j*StackMean_imag
            
#%% Singular Value decomposition and backazimuth calculation
            
            # Perform a singular value decomposition
            U,S,V=svd(StackMean,compute_uv=True)    
            
            # Conjugate transpose of V
            V=np.conj(V.T)
            
            #Save the first second and third singular value 
            s0.append(S[0]);s1.append(S[1]);s2.append(S[2])
        
            # Define the polarization vector
            Z=np.asarray(V[:,0])
            
            # Specify the components of polarization vector
            Z0=Z[0];Z1=Z[1];Z2=Z[2]
            
            # Change the polarization vector components to polar coordinates
            [r_z,phi_z]=cmath.polar(Z0)
            [r_x,phi_x]=cmath.polar(Z1)
            [r_y,phi_y]=cmath.polar(Z2)

            #calculate backazimuth  (in degrees)
            Az=np.arctan2(r_y,r_x)*360/(2*pi)
        
            # Determine which quadrant the backazimuth falls into based on imaginary values
            if Z1.imag >0:
                if Z2.imag>0:
                    Az=180+Az 
                else:
                    Az=180- Az
                    
            else:
                if Z2.imag >0:
                    Az=360-Az
                    
            
            # save backazimuth estimation                            
            Azimuth.append(Az)
            
            # Add 1 to split group count
            K=K+1
          
#%% Saving and binning backazimuth estimations
          
        # If the frequency is not dominated by Rayleigh waves record NAN
        if Frequency[x] not in GHT_freq_Rayleigh:
            AzBinned[LoopNumber,:]='Nan'
            PD[LoopNumber,:]='Nan'
            MaxPD[LoopNumber,:]='Nan'
            LoopNumber=LoopNumber+1
            continue
          
        # Histogram of backazimuth 
        AzBinned[LoopNumber,:],bins=np.histogram(Azimuth,bins)
        
        # Probability density of backazimuths in each bin
        PD[LoopNumber,:]=AzBinned[LoopNumber,:]/sum(AzBinned[LoopNumber,:])
        PD_tremor=PD
        
        # define the bins that have the largest probability density (with max number telling how many bins to saves)
        largest=heapq.nlargest(MaxNumber,range(len(PD[LoopNumber,:])),PD[LoopNumber,:].take)
        
        # Define the largest probability density bins
#        MaxBA=[]
#        for M in largest:    
#            MaxBA.append(bins[M])
#        MaxPD[LoopNumber,:]=MaxBA
        
#    with open('E:/Research/Results/%s/Backazimuths/BA_Combo.%s.pickle'%(Station,doy), 'wb') as f:  # Python 3: open(..., 'wb')
#        pickle.dump([Range,MaxPD,PD_tremor],f)
        
        # Add 1 to the loop count
        LoopNumber=LoopNumber+1
        
        
            
# %% Plotting backazimuth in rose diagram (like figure 4 of Vore et al., 2019)
            
#    WidthMask=ma.array(Width,mask=np.isnan(Width))
#    TremorWidthMask=ma.array(TremorWidth,mask=np.isnan(TremorWidth))
    
# replace probabilities that are less than 0.04 with 'NAN' for plotting clarity
    for j in np.arange(0,len(PD_tremor[0,:])):
        for i in np.arange(0,len(PD_tremor[:,0])):
            if PD_tremor[i,j] <0.04:
                PD_tremor[i,j]='Nan'
                
    # Masking values that are Nan 
    Zm = ma.array(PD_tremor,mask=np.isnan(PD_tremor)) 
    
    # Make bin 360 degrees into bin 0 degrees
    bins[-1]=0
    
    #Create same sized arrays for frequencies and bins
    freq=np.tile(freq,(len(bins),1)).T
    bins=np.tile(bins,(len(freq),1))
    
    # change bins from to radian 
    bins_rad=np.deg2rad(bins)
    
    gs = gridspec.GridSpec(1, 2,width_ratios=[10,1])
    
    # make a polar plot with color gradient legend
    ax1 = plt.subplot(gs[0], projection="polar", aspect=1.)
    ax2 = plt.subplot(gs[1])
    im=ax1.pcolormesh(bins_rad,freq,Zm,cmap='Reds',vmin=.04,vmax=0.22)
    ax1.set_rmax(Freqmin)
    ax1.set_rmin(Freqmax)
    ax1.set_rticks(np.arange(Freqmin,Freqmax,1))  
    ax1.grid(True,ls='solid')
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_title(str(day),size=16)
    cb1=plt.colorbar(im,cax=ax2)
    cb1.set_label('Probability of Backazimuth Location Per Frequency', rotation=270,labelpad=20)
    
    #save figure before continuing onto next day
    plt.savefig(str( analysis_path / 'Results' / 'PolarPlot_Video') + '/%s_Day%s'%(Station,doy))
    plt.close()
    

























    
       
        