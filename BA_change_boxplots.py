# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:58:24 2018
Calculates the median, 75th, 25th and all boxplot stats for the backazimuth measurements 
of bands of Rayleigh wave Glaciohydraulic tremor (GHT).This code is meant to run after  the frequenices
bands of GHT have been defined. This code is also used to plot boxplots and figure 6 in the paper
@author: vore5101
"""
#%% Packages Needed 
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from numpy.linalg import svd
import cmath
import pickle
from pathlib import Path
import configparser
import datetime as dt
from scipy import stats

#%% trendline function 
def trendline(xd, yd, order=1, c='r', alpha=1, Rval=False):
    """Make a line of best fit and calculates R^2 for the trendline"""

    #Calculate trendline
    coeffs = np.polyfit(xd, yd, order)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    power = coeffs[0] if order == 2 else 0

    minxd = np.min(xd)
    maxxd = np.max(xd)

    xl = np.array([minxd, maxxd])
    yl = power * xl ** 2 + slope * xl + intercept

    #Plot trendline
    plt.plot(xl, yl, c, alpha=alpha,color='r',linestyle='--')

    #Calculate R Squared
    p = np.poly1d(coeffs)

    ybar  = np.sum(yd) / len(yd)
    ssreg = np.sum((p(xd) - ybar) ** 2)
    sstot = np.sum((yd - ybar) ** 2)
    Rsqr  = ssreg / sstot

    if not Rval:
        #Plot R^2 value
        plt.text(235,55, '$R^2 = %0.2f$' % Rsqr)                            # USER: set y and x limits of text
        plt.text(235,60,'$y=%0.3fx + %0.3f$'%(slope,intercept))             # USER: set y and x limits of text
        return slope,intercept,Rsqr
    else:
        #Return the R^2 value:
        return Rsqr


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

# Freqmin = float(config['DEFAULT']['PolarFreqmin']) # Minimum frequancy of interest {Hz}
# Freqmax = float(config['DEFAULT']['PolarFreqmax']) # Max frequency of interest [Hz]
averaging_number = int(config['DEFAULT']['averaging_number']) # number of spectral covariance matricies that are averaged together

start_day = dt.datetime(year, start_month, start_day, 0,0,0)
end_day = dt.datetime(year, end_month, end_day, 0,0,0)

dates = np.arange(start_day, end_day, dt.timedelta(days=1))


#%% User defined variables 
# Station='ETIP'                      # Seismic Station Name
# year=2016                           # year of seismic data collection 
Freqmin=1.7                         # Minimum Frequency of GHT band [Hz]
Freqmax=3.3                         # Maximum frequency of GHT band [Hz]
# averaging_number=13                 # number of SCM that are averaged together
    
#%% 
# Initiate arrays 
boxplot_data=[]; label_doy=[];median_value=[]
upper_quartile=[]; lower_quartile=[]; doy_stats=[]


#%% Load Data Needed (found from GHT_Frequencies.py and FFT_loop.py)

Sin0=[];Sin1=[];Sin2=[];SinTot=[]   

#Loops through all days of year in day range      
for day in dates:
    temp = (day - np.datetime64(str(year-1) + '-12-31')).astype('timedelta64[D]')                                            # defines the day of year
    doy = int( temp / np.timedelta64(1, 'D') )    

    #where frequencies of Rayleigh wave tremor data is stored 
    A= str( analysis_path / 'Results' / Station / 'freqs_by_WT'/'Rayleigh' ) + '/freqRayleigh_%s.pickle'%(doy)

    try:
        #load Rayleigh wave glaciohydraulic tremor frequencies
        with open(A,'rb') as f:
            GHT_freq_Rayleigh= pickle.load(f)
        GHT_freq_Rayleigh=GHT_freq_Rayleigh[0]
        
        # Load fft results for day of year (as found by FFT_loop.py)
        with open(str( analysis_path / 'FFT_results' / Station ) + '/FFT.%s.pickle'%(doy),'rb') as f:  # Python 3: open(..., 'rb')
            FFT_stack_real,FFT_stack_imag,WindowLoop,Frequency,averaging_number= pickle.load(f)    
    
    #If a given day of data doesn't exist record a zero in boxplot_data array
    except IOError:
        boxplot_data.append([0])
        label_doy.append(doy)
        continue    
    
    # determine if there are any frequencies that are GHT Rayleigh waves within the frequency range for given day
    GHT_Freq_Range=[ x for x in GHT_freq_Rayleigh if (x >=Freqmin and x <=Freqmax)]
   
   # if there are no GHT Rayleigh waves within the frequency range, record a zero in boxplot_data and continue loop
    if len(GHT_Freq_Range)==0:
        boxplot_data.append([0])
        label_doy.append(doy)
        continue
    
    # Find the index values from FFT frequencies that coinside with the frequency range of tremor for a given day
    index=[]
    for B in GHT_Freq_Range:
       index.append(np.where(Frequency==B)[0][0])
    # Array of the index values 
    Range=np.arange(min(index),max(index)+1)
    
    #Initiate array
    Azimuth=[]
    
    # Split FFT into groups of user defined size to be able to average out short transient seismic events
    Split=np.split(WindowLoop, len(WindowLoop)/float(averaging_number))
    LoopNumber=0

#%% Calculating backazimuth measurements    
    # x in the range of frequency values
    for x in Range : 
        
        # If the frequency is not in the GHT Rayleigh wave frequencies the frequency is skipped
        if Frequency[x] not in GHT_Freq_Range:
            continue
        
        # Initiate group counter
        K=0
        
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
                    
#%% Singular Value Decomposition 

            # perform singular value decomposition on spectral covarience matrix
            U,S,V=svd(StackMean,compute_uv=True)    
            
            # Get transpose of V
            V=np.conj(V.T)
        
            # polarization vector defined along with each component 
            Z=V[:,0]
            Z=np.asarray(Z); Z0=Z[0];Z1=Z[1];Z2=Z[2]

            
            XY=Z[1]**2+Z[2]**2
            
            # change all components of polarization vector into polar coordianates
            [r_z,phi_z]=cmath.polar(Z0)
            [r_x,phi_x]=cmath.polar(Z1)
            [r_y,phi_y]=cmath.polar(Z2)
            [r_xy,phi_xy]=cmath.polar(XY)
            
#%% Calculate backazimuth 
            
            Az=np.arctan2(r_y,r_x)*360/(2*pi)
        
            # Assign backazimuth to appropriate quadrant 
            if Z1.imag >0:
                if Z2.imag>0:
                    Az=180+Az 
                else:
                    Az=180- Az
                    
            else:
                if Z2.imag >0:
                    Az=360-Az
            
            # Save backazimuth measurement               
            Azimuth.append(Az)
            
            # Add one to group counter
            K=K+1
        
#%% Finding Statistics of backazimuth results  
    mean_az = stats.circmean(Azimuth, high=360, low=0)
    delta_az = (mean_az-180)%360
    Az_off = (Azimuth - delta_az)%360 # azimuth offset

    #median of backazimuths 
    median_value.append( (np.median(Az_off) + delta_az)%360 )
    
    #75th percentile of backazimuths
    upper_quartile.append( (np.percentile(Az_off, 75) + delta_az)%360 )
    
    #25 percentile of backazimuths
    lower_quartile.append( (np.percentile(Az_off, 25) + delta_az)%360 )
    
    #day of year
    doy_stats.append(doy)
    
    # Backazimuth array
    boxplot_data.append(Az_off)
    
    # poitions for labels 
    label_doy.append(doy)
    
# Save data 
with open(str( analysis_path / 'TimeSeries') + '/%s_%s_%s.pickle'%(Station,Freqmin,Freqmax), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([median_value,upper_quartile, lower_quartile,boxplot_data,label_doy,doy_stats], f)

#%% Used to make boxplots of data (done originally for figure 6)
# with open(str( analysis_path / 'TimeSeries') + '/BBEU_1.7_3.3.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
#         median_value,upper_quartile, lower_quartile,boxplot_data,label_doy,doy_stats=pickle.load(f)
     
bp_dict =plt.boxplot(boxplot_data,positions=label_doy,showfliers=False)
plt.xlabel('DOY')
plt.ylabel('Backazimuth [degrees]') 
# plt.ylim(205,265)
# plt.xlim(xmax=195)
plt.title('Temporal Changes in Backazimuth Measurements:%s to %s'%(Freqmin,Freqmax))      
plt.savefig(str( analysis_path / 'Results') + '/Boxplots_'+Station)
## Place Trendline through the median backazimuth for analysis purposes    
#slope,intercept,Rsqr=trendline(doy_stats,median_value) 
#plt.ylim(50,270)  

#%%
fig,ax=plt.subplots()
ax.fill_between(dates,upper_quartile,lower_quartile,color='purple',alpha=0.2)
ax.errorbar(dates,median_value, yerr=[np.array(median_value)-np.array(lower_quartile),np.array(upper_quartile)-np.array(median_value)],fmt='o',color='purple',label='3.3-4.0 Hz')
# ax.fill_between(np.array(doy_stats2)+.35,upper_quartile2,lower_quartile2,color='red',alpha=0.2)
# ax.errorbar(np.array(doy_stats2)+.35,median_value2, yerr=[ np.array(median_value2)-np.array(lower_quartile2),np.array(upper_quartile2)-np.array(median_value2)],fmt='o',color='r',label='4.0-4.5 Hz')
ax.set_ylabel('Backazimuth [degree]')
ax.legend(loc=2,fontsize='medium')
ax.set_title(Station)
plt.savefig(str( analysis_path / 'Results') + '/Errorbars_'+Station)


#%% Used to make figure 6 in paper

# # Open different frequency data sets 
# with open('D:/Research/TimeSeries/ETIP_3.3_4.0.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
#         median_value1,upper_quartile1, lower_quartile1,boxplot_data1,label_doy1,doy_stats1=pickle.load(f)
# with open('D:/Research/TimeSeries/ETIP_4.0_4.5.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
#         median_value2,upper_quartile2, lower_quartile2,boxplot_data2,label_doy2,doy_stats2=pickle.load(f)

# fig,ax=plt.subplots()
# ax.fill_between(doy_stats1,upper_quartile1,lower_quartile1,color='purple',alpha=0.2)
# ax.errorbar(doy_stats1,median_value1, yerr=[np.array(median_value1)-np.array(lower_quartile1),np.array(upper_quartile1)-np.array(median_value1)],fmt='o',color='purple',label='3.3-4.0 Hz')

# ax.fill_between(np.array(doy_stats2)+.35,upper_quartile2,lower_quartile2,color='red',alpha=0.2)
# ax.errorbar(np.array(doy_stats2)+.35,median_value2, yerr=[ np.array(median_value2)-np.array(lower_quartile2),np.array(upper_quartile2)-np.array(median_value2)],fmt='o',color='r',label='4.0-4.5 Hz')

# ax.set_ylabel('Backazimuth [degree]')
# ax.legend(loc=2,fontsize='medium')
# ax.set_title('ETIP')

        

