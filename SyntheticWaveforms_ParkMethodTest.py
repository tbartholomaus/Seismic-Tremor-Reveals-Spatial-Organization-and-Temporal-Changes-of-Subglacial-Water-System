# -*- coding: utf-8 -*-
"""

Created on Tue Jul 11 11:15:41 2017
@author: Margot E. Vore

This program creates sythetic waveforms to test the wave polarization technique 
that is outlined in Park et al. 1987

"""
#%% Packages Needed

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from numpy.fft import fftfreq 
from numpy.linalg import svd
import cmath
from math import pi,cos,sin
from itertools import islice
from spectrum import*

#%%  function that creates the subsets of the waveform (windows)   
def window(seq, n=2):       
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# %% User Defined Variables of Synthetic waveform 
loop=0
Avg=[]
Freq_experiment=[]


VertA       = 1                                                                 # Vertical Amplitude
NorthA      = 1                                                                 # North Amplitude
EastA       = -1                                                                # East Amplitude
expected    = np.mod( np.arctan2(EastA, NorthA)/(2*pi)*360 , 360)               # degrees, clockwise of north
freq_exp    = 7                                                                 # frequency of waveform
Lag         = pi/2                                                              # Lag of vertical to horizontal channels  
Noise       = 'yes'                                                             # 'yes' or 'no'   Turns noise on and off
Noise_C     = 6                                                                 # Scales the noise
sample_rate = 0.01                                                              # sampling rate of your waveform 
wav_length  = 50                                                                # [sec]  Length of the waveform
MaxFreq     = 10                                                                # Frequency to stop code at

# %% Creation of waveform 

T = np.arange(0, wav_length, sample_rate)

if Noise=='yes':                                                                #creating wavefom with noise
    R=np.random.randn(len(T),3)                                                 #creates the normally distributed noise array to add to data
    Vert  = VertA *sin(2*pi*freq_exp*T+Lag)+Noise_C*R[:,0] 
    North = NorthA*sin(2*pi*freq_exp*T)+Noise_C*R[:,1]
    East  = EastA *sin(2*pi*freq_exp*T)+Noise_C*R[:,2]


else:                                                                           # creates waveform without noise            
    Vert  = VertA *sin(2*pi*freq_exp*T+Lag)
    North = NorthA*sin(2*pi*freq_exp*T)
    East  = EastA *sin(2*pi*freq_exp*T)

# %% Plotting the Waveforms

#fig=plt.figure("Waveforms")
#
#
#ax1= fig.add_subplot(311)
#ax2= fig.add_subplot(312)
#ax3= fig.add_subplot(313)
#
#
#ax1.plot(T, Vert,  'b',label='Vertical')                                        # Plots Vertical waveform
#ax1.tick_params(axis='x',which='both', bottom='on' ,top='off', labelbottom='off')
#ax1.set_xlim([0,1])
#ax1.set_ylim([-20,20])
#ax1.set_ylabel('Vertical')
#
#
#ax2.plot(T, North, 'b',label='North')                                           # Plots North Waveform
#ax2.tick_params(axis='x',which='both', bottom='on',top='off', labelbottom='off')
#ax2.set_xlim([0,1])
#ax2.set_ylim([-20,20])
#ax2.set_ylabel('North')
#
#
#ax3.plot(T, East,  'b',label='East')                                            # Plots East waveform
#ax3.set_xlim([0,1])
#ax3.set_ylim([-20,20])
#ax3.set_ylabel('East')
#ax3.set_xlabel('Time [s]')

# %% Creates a sliding window of in order to split waveform into subsets

win_len = 5                                                                     # USER: [sec] defines window width 
width = int(win_len/sample_rate)                                                # number of samples per window


Vwin=[];Nwin=[];Ewin=[]                                                         # splitting the waveform up into subsets (windows)    


for windowed_Vert in window(Vert,n=width):                                      # Vertical Subsets
    windowed_Vert=windowed_Vert
    Vwin.append(windowed_Vert) 


for windowed_North in window(North,n=width):                                    # North Subsets
    windowed_North=windowed_North
    Nwin.append(windowed_North) 

    
for windowed_East in window(East,n=width):                                      # East Subsets
    windowed_East=windowed_East
    Ewin.append(windowed_East) 

# %% Prepare for the FFT

StartWin=0
EndWin         = len(Ewin)
WindowLoop     = np.arange(StartWin,EndWin,1)


FFT_stack_real = np.empty((len(Vwin[0]),3,len(WindowLoop)))                     # initialing of arrays where the FFT results will be stacked for analysis later on
FFT_stack_imag = np.empty((len(Vwin[0]),3,len(WindowLoop)))
stacked_real   = np.empty((3,3,len(WindowLoop)))
stacked_imag   = np.empty((3,3,len(WindowLoop)))


NW=2.5                                                                          # USER: defines the number of tapers we are creating NW*2 
[tapers,eigen]=dpss(len(Vwin[0]),NW)                                            # defining the eigentapers used for the tapering    


eigen  = np.delete(eigen,  np.where (eigen<0.90)[0])                            # remove eigentapers if corresponding eigenvalue is less than .90
tapers = np.delete(tapers, np.where (eigen<0.90)[0],axis=1)


FFT_Z_real = np.empty((len(Vwin[0]),len(eigen)))                                # initiating arrays that will be used for averaging after the taper is applied
FFT_N_real = np.empty((len(Vwin[0]),len(eigen)))
FFT_E_real = np.empty((len(Vwin[0]),len(eigen)))
FFT_Z_imag = np.empty((len(Vwin[0]),len(eigen)))
FFT_N_imag = np.empty((len(Vwin[0]),len(eigen)))
FFT_E_imag = np.empty((len(Vwin[0]),len(eigen)))



#%% Perfoming the FFT using a prolate spherical taper on subwindows

for i in WindowLoop:                                                            # runs for all subwindows


        Frequency = fftfreq(width, d=sample_rate)
        for tap_num in np.arange(0,len(eigen),1):
            FFT_Z                 = fft(tapers[:,tap_num]*Vwin[i])/len(Vwin[i]) #fft on vertical
            FFT_Z_real[:,tap_num] = FFT_Z.real                                  #Saves real and imaginary seperatly 
            FFT_Z_imag[:,tap_num] = FFT_Z.imag
            FFT_N                 = fft(tapers[:,tap_num]*Nwin[i])/len(Nwin[i]) # fft on north
            FFT_N_real[:,tap_num] = FFT_N.real
            FFT_N_imag[:,tap_num] = FFT_N.imag
            FFT_E                 = fft(tapers[:,tap_num]*Ewin[i])/len(Ewin[i]) # fft on east
            FFT_E_real[:,tap_num] = FFT_E.real
            FFT_E_imag[:,tap_num] = FFT_E.imag


        FFT = np.mean(FFT_Z_real,1) + 1j*np.mean(FFT_Z_imag,1)                  # mean of the multi-tapered spectra
        FFT = np.c_[ FFT, np.mean(FFT_N_real,1) + 1j*np.mean(FFT_N_imag,1) ]    # Concatenate the other channels with the Z channel
        FFT = np.c_[ FFT, np.mean(FFT_E_real,1) + 1j*np.mean(FFT_E_imag,1) ]


        FFT = np.asmatrix(FFT)                                                  # The complete FFT for window i


        FFT_stack_real[:,:,i]=FFT.real                                          # splitting up the real and imaginary parts of the FFT for each window and saving results
        FFT_stack_imag[:,:,i]=FFT.imag

# %% Create Spectral Covariance Matrix (SCM) and perform Singular Value Decomposition (SVD)

Phi=[]; Azimuth_sig=[]; freq_sig=[]; s0=[]; s1=[]; s2=[]                        # Initate arrays where results will be placed
Azimuth_nonsig=[]; freq_nonsig=[]


for x in np.arange(0, len(Frequency), 1):                                       # loops through frequencies


    for i in WindowLoop: #len(Vwin)                                             # loops through  windows
        FFT_working = FFT_stack_real[x, :, i] + 1j*FFT_stack_imag[x, :, i]      
        FFT_working = np.asmatrix(FFT_working)
        FFT_T       = np.conj(FFT_working.T)
        SpecCov     = np.dot(FFT_T, FFT_working)                                # compute spectral covariance matrix (SCM) for a gievn frequency
       
       
        stacked_real[:,:,i] = SpecCov.real                                      #Save SCM
        stacked_imag[:,:,i] = SpecCov.imag 
    
    
    StackMean_real = np.mean(stacked_real,2)                                    # linearly average all SCM for a gievn frequency [ creates SCM for whole waveform]
    StackMean_imag = np.mean(stacked_imag,2)
    StackMean      = StackMean_real + 1j*StackMean_imag
    
    
    U,S,V=svd(StackMean,compute_uv=True)                                        # perform SVD on averaged SCM
    V=np.conj(V.T)                                                              # take conjugate transpose of V from SVD
    s0+=[S[0]]; s1+=[S[1]]; s2+=[S[2]]                                          # record singular values
    
    
    Z=V[:,0]                                                                    # create the polarization vector Z and specify components
    Z=np.asarray(Z); 
    Z0=Z[0];Z1=Z[1];Z2=Z[2]
    z_2=Z[2].real; z_1=Z[1].real
    
 
#%% Calculation of lag between horzontal and vertical channel   
       
    XY=Z[1]**2+Z[2]**2
    
    [r_z,phi_z] = cmath.polar(Z0)                                               # change polarization vector into polar coordiantes with amplitude and phase info
    [r_x,phi_x] = cmath.polar(Z1) 
    [r_y,phi_y] = cmath.polar(Z2) 
    [r_xy,phi_xy]=cmath.polar(XY)
    
    
    a_H=[]                                                                      # Initiate variable
    integer=np.arange(-1,2,1)
    
    
    for l in integer:                                                           # loop finds the major axis of horizonal motion eclipse to determine the horizontal phase angle
        Phi_H=-0.5*(phi_xy)+(l*pi)/2
        a_H.append((r_x**2*cos(Phi_H+phi_x)**2)+(r_y**2*cos(Phi_H+phi_y)**2))
    aH_max=np.where(a_H==max(np.asarray(a_H)))[0]
    l=integer[aH_max[0]]
    
    
    Phi_H=-0.5*(phi_xy)+(l*pi)/2                                                # Horizontal Phase lag
    
    
    Phi_VH=(Phi_H-phi_z)*(180/pi)-180                                           #phase lag between horizontal and vertical motion
    Phi.append(Phi_VH)
    
#%% Calculate the backazimuth of the waveform
    
    
    Az=np.arctan2(r_y,r_x)*360/(2*pi)                                           # Calculates Backazimuth 
    
    if Z1.imag >0:                                                              # Determines the quadrent the backazimuth falls in
        if Z2.imag>0:
            Az=180+Az
        else:
            Az=180- Az       
    else:
        if Z2.imag >0:
            Az=360-Az
    
    
    if S[0]>S[1]*5:                                                             #USER: Determines if ratio between first and second singular value is significant
        Azimuth_sig+=[Az]
        freq_sig.append(Frequency[x])
    else:
        Azimuth_nonsig+=[Az]
        freq_nonsig.append(Frequency[x])


    if Frequency[x]> MaxFreq:                                                    # stop loop when frequency exceeds a maximum frequency
        break
Avg.append(Azimuth_sig)
Freq_experiment.append(freq_sig)

#%% Plotting

plt.figure("BackAz v Freq")
plt.clf()


plt.scatter(freq_sig,Azimuth_sig,s=40,color='b',label='Significant Singular Values')    # Plots significant backazimuth measurements
plt.scatter(freq_nonsig,Azimuth_nonsig,s=40,facecolor='none',edgecolor='k',label='Nonsignificant Singular Values') # plots non-significant singular values


plt.axhline(expected,color='r',lw=0.7)                                          # creates a horizontal line at the expected backazimuth measurement


plt.axvline(freq_exp,color='r',lw=0.7)                                          # creates a vertical line at the expected frequency


plt.ylim(ymin=0,ymax=360)                                                       # USER: Change parameters of plots as needed
plt.xlim(xmin=0,xmax=10)
plt.ylabel('Backazimuth [degrees]')
plt.xlabel('Frequency [Hz]')
plt.title('Synthetic example' )
plt.legend(loc='lower left',fontsize='medium')
plt.plot()



#WHERE THE HORIZONTAL AND VERTICAL LINE INTERSECT IS WHERE THE BACKAZIMUTH IS EXPECTED TO BE FOR THE FREQUENCY OF THE WAVEFORM. 

#NOTE: Each time the a new waveform with noise is made, the noise changes which changes the backazimuth results.









