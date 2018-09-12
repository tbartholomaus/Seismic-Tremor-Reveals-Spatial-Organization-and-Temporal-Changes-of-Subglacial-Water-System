# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:41:28 2017
This script finds the power values of the seismic data that can be considered peaks in power 
values as constrained by the findings of the Power_Constraint.py script. The code finds averages
the power values for a range of frequencies on a given day, determines the local min and max values within the power data set
and finds the power values that are considered peak power. 


@author: vore5101
"""
#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime
import pickle
import datetime
from scipy.signal import argrelextrema as relex

#%% function that smooths the data set (y) between given number of points (box_pts)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    
#%% User defined variables
Station= 'GIW3'                 #Station
year=2016                       # year of data collection 
month=7                         # month of data collection 
day_range=np.arange(1,32,1)     #day of the month 
hr_s=00                         # starting hour of data 
min_s=00                        # starting min of data
sec_s= 00                       # starting sec of data
Freqmin=1.5                     # minimum frequency of data
Freqmax=10                      # maximum frequency of data
smoothing_number=5              # how many points are smoothed together in power data

#%% Power constraints as found from the Power_Constraint.py code

# Each station has a different power constraint due to different power levels
Power_constraint={'RTBD': 0.34,
                  'TWLV': 0.32,
                  'ETIP': 1.33,
                  'GIW1': 0.33,
                  'GIW2': 0.31,
                  'GIW3': 0.25,
                  'GIW4': 0.45,
                  'GIW5': 0.25,
                  'BBEU': 0.27,
                  'BBEL': 0.17,
                  'BBWU': 0.23,
                  'BBWL': 0.20}
                  

#%% lower limit constraints as found from the Power_Cobstraint.py code 

# Will cut out power peaks that are less than these power values                 
LL_constraint={'RTBD': 0.056,
               'TWLV': 0.051,
               'ETIP': 0.23,
               'GIW1': 0.067,
               'GIW2': 0.065,
               'GIW3': 0.047,
               'GIW4': 0.086,
               'GIW5': 0.062,
               'BBEU': 0.028,
               'BBEL': 0.028,
               'BBWU': 0.033,
               'BBWL': 0.026}
#%%    
# Open PSD of given station 
with open('E:/Research/PowerSpectralDensity/mp%s.pickle'%Station, 'rb') as f:  
            t, freqsPower, Pdb_array, pp, data_dir, station_PSD = pickle.load(f)
            
# loops through each day in the day range specified by user           
for day in day_range:
    
    dt=datetime.datetime(year,month,day, hr_s,min_s,sec_s)  # convert to datetime 
    dayofyear=int(dt.strftime('%j'))                        # find the day of year
    
    # find the index values that fall between the minimum and maximum frequencies
    freq1=np.where((freqsPower>Freqmin))[0][0]
    freq10=np.where((freqsPower>Freqmax))[0][0]
    freqsPower_cut=np.delete(freqsPower[freq1:freq10],0)[:-1] 
    
    # define the indicies that contain a days worth of power (can be changed to different time scales of desiered)
    starting_index=np.where(t==UTCDateTime(year,month,day,hr_s,min_s))[0][0] # start of day indicie
    # end of day indice    
    try:
        ending_index= np.where(t==UTCDateTime(year,month,day+1,hr_s,min_s))[0][0] 
    except ValueError:
        ending_index= np.where(t==UTCDateTime(year,month+1,1,hr_s,min_s))[0][0]
    time=t[starting_index:ending_index] # time span of interest
    
    # Find the average power between min and max frequency for each time index
    Power=np.zeros((len(time),len(freqsPower))) # initiate power array
    # take power from the PSD array for all frequencies of interest for a given time
    for x in np.arange(starting_index,ending_index,1):
        Power[x-starting_index,:]=Pdb_array[:,x]
    # find averae power between frequency containts for each time
    Power_average=np.median(Power, axis=0)[freq1:freq10]
    
    # smooth the power average for ease of finding local min and max values
    Power_average=np.delete(smooth(Power_average,smoothing_number),0)[:-1]
 
#%% Find local minimum and maxiumum power values for a given day 
   
    # find indicies that define the location of a local min or max value in the power
    locmin_index=relex(Power_average,np.less)[0]
    locmax_index=relex(Power_average,np.greater)[0]
    
    # make each local extrama arrays start with a min value for ease of use
    try:
        if locmax_index[0] < locmin_index[0]:
            locmin_index= np.insert(locmin_index,0,0)
    except IndexError:
        continue
    
    # define the local min and max power values from the index 
    locmin_value=np.take(Power_average,locmin_index)
    locmax_value=np.take(Power_average,locmax_index)
    
#%% remove local min/max values if the differnece between the two is less than the 20th percentile of differences 
        # 20th percentile is the LL_constraints as defined in Power_constraint.py
    R=[]
    # find the indicies that fall below the 20th percentile of differences
    for x in np.arange(0,len(locmin_value)-1,1):
        if locmax_value[x] -locmin_value[x] < LL_constraint[Station]:
            R.append(x) 
        if locmax_value[x]-locmin_value[x+1] < LL_constraint[Station]:
            R.append(x+1)
    
    # define which indecies should be removed (withour repeating any indicies)
    remove=[]
    for i in R:
      if i not in remove:
        remove.append(i)
    # delete indicies with insignificant differences between local extrema 
    locmin_index=np.delete(locmin_index,remove)
    locmin_value=np.delete(locmin_value, remove)
    
#%% find frequencies that contain a power peak 
    PP=[]
    for P in np.arange(0,len(locmin_index)-1,1):
        Power_use=Power_average[locmin_index[P]:locmin_index[P+1]] # specify the indicies between the local extream
        # foe each index in power use, find which are greater than both its surrounding local minimum values by more than the power constraint of the station 
        PP.append(np.where(np.logical_and(np.greater_equal(np.array(Power_use),locmin_value[P]+Power_constraint[Station]),np.greater_equal(np.array(Power_use), locmin_value[P+1]+Power_constraint[Station])))[0]+locmin_index[P])
    # indicies that are power peaks 
    Peak_Power=np.concatenate(PP)
    
    # finds the frequencies assoicated with the power peak indicies
    PeakPower_freq=np.take(freqsPower_cut,Peak_Power)
    
    # finds the values associated with the power peak indicies
    PeakPower_values=np.take(Power_average,Peak_Power)
    
    # Save the power peak frequencies and power values
    with open('E:/Research/Constraints/%s/Power/PP.%s.pickle'%(Station,dayofyear), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([freqsPower_cut[1:-1],Power_average[1:-1],PeakPower_freq,PeakPower_values ],f)

#%% Plotting the power peaks for a given day 

#    with open('E:/Research/Constraints/%s/Power/PP.%s.pickle'%(Station,dayofyear), 'rb') as f:  # Python 3: open(..., 'rb')
#        freqsPower_cut,Power_average,PeakPower_freq,PeakPower_values = pickle.load(f)
#    f_min=np.take(freqsPower_cut,locmin_index)
#    plt.plot(freqsPower_cut,Power_average)
#    plt.scatter(f_min,locmin_value)
#    plt.scatter(PeakPower_freq,PeakPower_values,color='r')
#    plt.ylabel('Power')
#    plt.xlabel('Frequency')
#    plt.title('%s: day %s- Power Peaks'%(Station,dayofyear))