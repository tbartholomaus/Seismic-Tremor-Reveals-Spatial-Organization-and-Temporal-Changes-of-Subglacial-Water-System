# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:01:54 2017
This scripts finds the frequencies of a seismic signal that contains peaks in the ratio between 
the first and second singular values as well as the distribution of wave types (Rayleigh, body, and mixed)
within each frequency band for a given day. 
@author: vore5101
"""
#%% Packages Needed  
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from numpy.linalg import svd
import cmath
from math import cos
import pickle
import datetime

#%% function that smooths the data set (y) between given number of points (box_pts)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth 
    
#%% User defined variables
Station='GIW3'                  # Station Name
year=2016                       # year of data collection
month=7                         # month of interest within data set
day_range=np.arange(1,32,1)     # day range of interest within data set 
hr_s=00                         # starting hour of data of interest 
min_s=00                        # starting minute of data of interest
sec_s= 00                       # starting second of data of interest
SV_constraint=2.5               # Singular value threshold value
WT_constraint=39                # percent of wavetype threshold value
WT_Other_constraint=96          # percent of mixed wavetype threshold value
Freqmin=1.5                     # minimum frequency of interest [Hz]
Freqmax=10                      # maximum frequency of interest [Hz]

#%% finds the singular value and wave type peaks for each day in the seismic record
for day in day_range:
    
    dt=datetime.datetime(year,month,day, hr_s,min_s,sec_s) # date into datetime format
    dayofyear=int(dt.strftime('%j')) # determines the day of year
    
    # opens FFT results for gievn day (From FFT_loop.py)
    try:
        with open('E:/FFT_results/%s/FFT.%s.pickle'%(Station,dayofyear),'rb') as f:  
                FFT_stack_real,FFT_stack_imag,WindowLoop,Frequency,averaging_number= pickle.load(f)    
    except IOError:
        continue
    
    # find inidice of the min and max frequency 
    FreqMin_location=np.where(np.array(Frequency)>Freqmin)[0]
    FreqMax_location=np.where(np.array(Frequency)>Freqmax)[0] 
    
    # Initiate arrays 
    Sin0=[];Sin1=[];Sin2=[]  # Singular values
    freq=[]; # frequency
    Phi020_O=[];PhiOther_O=[];PhiRayleigh_O=[]  # waveform type
    
    # Split the time window into groups for averaging purposes 
        # remove short term events from affecting the GHT signal 
    Split=np.split(WindowLoop, len(WindowLoop)/float(averaging_number))
    
    # For each frequency
    for x in np.arange(FreqMin_location[0],FreqMax_location[0]+1,1): 
        
        freq.append(Frequency[x])
        Phi=[]
        K=0 # loops throuh the number of split windows
        s0=[];s1=[];s2=[]
        
        # for each time group within a single frequency 
        while K <len(Split): 
            
            # choose a subset of windows to work with 
            SplitLoop=Split[K] 
            stacked_real=np.empty((3,3,len(SplitLoop)))
            stacked_imag=np.empty((3,3,len(SplitLoop)))
 
#%% Calculate average spectral covarience matrix           
            # Create spectral covarience matrix for each time in a group
            for i in SplitLoop: 
                # recombine imaginary and real parts
                FFT_working=FFT_stack_real[:,:,i]+1j*FFT_stack_imag[:,:,i]
                FFT_working=np.asmatrix(FFT_working)
                
                # Get conjugate transpose of the FFT 
                FFT_T=np.conj(FFT_working[x].T)
                
                # get spectral covarience matrix and split into real and imaginary parts
                SpecCov=np.dot(FFT_T,FFT_working[x])
                stacked_real[:,:,i-SplitLoop[0]]=SpecCov.real
                stacked_imag[:,:,i-SplitLoop[0]]=SpecCov.imag
            
            # get the average spectral covarience matrix of all times in the group
            StackMean_real=np.mean(stacked_real,2)
            StackMean_imag=np.mean(stacked_imag,2)
            StackMean=StackMean_real+1j*StackMean_imag
            
#%% SVD and polarization vector
            # perform singular value decomposition on spectral covarience matrix
            U,S,V=svd(StackMean,compute_uv=True)    
            
            # Get transpose of V
            V=np.conj(V.T)
            
            # record singular values that are calculated by singular value decomposition 
            s0.append(S[0]);s1.append(S[1]);s2.append(S[2])
        
            # polarization vector defined along with each component 
            Z=V[:,0]
            Z=np.asarray(Z); Z0=Z[0];Z1=Z[1];Z2=Z[2]
            
            XY=Z[1]**2+Z[2]**2
            
            # change all components of polarization vector into polar coordianates
            [r_z,phi_z]=cmath.polar(Z0)
            [r_x,phi_x]=cmath.polar(Z1)
            [r_y,phi_y]=cmath.polar(Z2)
            [r_xy,phi_xy]=cmath.polar(XY)

#%% calculate the backazimuth of the seismic signal for each group    

            # conditions for determining the the horizontal angle of the seismic signal 
                # see park et al. (1987)        
            i=0
            a_H=[];integer=np.arange(-1,2,1)
            for l in integer: #l has to be an integer
                Phi_H=-0.5*(phi_xy)+(l*pi)/2
                a_H.append((r_x**2*cos(Phi_H+phi_x)**2)+(r_y**2*cos(Phi_H+phi_y)**2))
            a_H_max=np.where(np.asarray(a_H)==max(a_H))[0]
            l=integer[a_H_max[0]]  
            
            #calculate the horizontal angle
            Phi_H=-0.5*(phi_xy)+(l*pi)/2   
            
            #calculate backazimuth and assign to appropriate quadrant 
            Phi_VH=(Phi_H-phi_z)*(180/pi)
            if Phi_VH > 90:
                Phi_VH=Phi_VH-180
            elif Phi_VH <-90:
                Phi_VH=Phi_VH +180
            Phi.append(Phi_VH) 
            
            # move to next grouping 
            K=K+1
        
        # Find the average singluar values for all the groupings 
        Sin0.append(np.median(s0));Sin1.append(np.median(s1));Sin2.append(np.median(s2))
        
#%% Find percentage of wave types within one day at one frequency
            # counting how many of occurances of each wave type there is
        A=0;B=0;C=0
        for x in Phi:
            # defines a body wave
            if (x>=0 and x<20) or (x<0 and x>-20):
                A=A+1
            
            # defines a mixed wave
            elif (x>=20 and x <70) or (x<=-20 and x>-70):
                B=B+1
            
            #defines a Rayleigh wave
            elif  x>=70 or x<-70:
                C=C+1
                
        # percentage of wave type 
        Phi020_O.append(float(A)/float(len(Phi))*100)
        PhiOther_O.append(float(B)/float(len(Phi))*100)
        PhiRayleigh_O.append(float(C)/float(len(Phi))*100.0)
        
    # Smooth the wave type percentages using the smoothing function for easier interpretation     
    Phi020=np.delete(smooth(Phi020_O,4),0)[:-1]
    PhiRayleigh=np.delete(smooth(PhiRayleigh_O,4),0)[:-1]
    PhiOther=np.delete(smooth(PhiOther_O,4),0)[:-1]

#%% Define frequency and values of peaks in Singular Values and wave type
    freq=freq[1:-1]
    
    # finding the SV ratio between 1st and 2nd singular values
    SV_raw=np.array(Sin0)/np.array(Sin1)
    SV= smooth(SV_raw,8)[1:-1]
    
    # Index, value and frequency of peak SV ratio values 
    SVPeak_index=np.where(np.array(SV)>SV_constraint)[0]
    SVPeak_values=np.take(SV,SVPeak_index)
    SVPeak_freq=np.take(freq,SVPeak_index)
    
    # Find index where each wave type percentage exceeds the threshold value
    RPeak_index= np.where(np.array(PhiRayleigh)> WT_constraint)[0]
    BPeak_index=np.where(np.array(Phi020)> WT_constraint)[0]
    OPeak_index=np.where(np.array(PhiOther)>WT_Other_constraint)[0]
    
    # find the values of the wave type peaks
    RPeak_values=np.take(PhiRayleigh,RPeak_index)
    BPeak_values=np.take(Phi020,BPeak_index)
    OPeak_values=np.take(PhiOther,OPeak_index)
    
    # Find the frequencies of wave type peaks 
    RPeak_freq=np.take(freq,RPeak_index)
    BPeak_freq=np.take(freq,BPeak_index)
    OPeak_freq=np.take(freq,OPeak_index)

#%% Save results
    
    # Singular Value
    with open('E:/Research/Constraints/%s/SV/SVP.%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([freq, SV, SVPeak_values, SVPeak_freq],f)

    # Rayleigh wave
    with open('E:/Research/Constraints/%s/Rayleigh/RP.%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([freq, PhiRayleigh, RPeak_values, RPeak_freq],f)
    
    # Body wave 
    with open('E:/Research/Constraints/%s/020/ZP.%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([freq, Phi020, BPeak_values, BPeak_freq],f)
    
    # Mixed wave
    with open('E:/Research/Constraints/%s/Other/OP.%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([freq, PhiOther , OPeak_values, OPeak_freq],f)
        
#%% Plotting
    plt.plot(freq,SV)
    plt.scatter(SVPeak_freq,SVPeak_values,color='g')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('S0/S1')
    plt.title('%s Day %s: Peaks in Singular values'%(Station, dayofyear))
        
    plt.plot(freq,PhiRayleigh)
    plt.scatter(RPeak_freq,RPeak_values,color='g')
    
    plt.plot(freq,Phi020)
    plt.scatter(BPeak_freq,BPeak_values,color='g')
        
    plt.plot(freq,PhiOther)
    plt.scatter(OPeak_freq,OPeak_values,color='g')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
