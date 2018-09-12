# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 11:23:26 2017
Used to define the power threshold for each seismic station by defining the lower limit (LL_constraint) and the
power constraint that defines the threshold value to determine what is considered a peak in seismic power. 
@author: vore5101
"""
#%% Packages
import numpy as np
from obspy.core import UTCDateTime
import pickle
import datetime
from scipy.signal import argrelextrema as relex
#---------------------------------------------------------------
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    
#%% User Defined Variables
Station= 'RTBD'
year=2016
month=7
day_range=np.arange(1,2,1)     #day of the month
hr_s=00
min_s=00
sec_s= 00
Freqmin=1.5
Freqmax=10
smoothing_number=5  # Number of points used for smoothing the power 

#%%

Distance=[]

#Upload power spectral density
with open('mpZQ_%s.pickle'%Station, 'rb') as f:  # Python 3: open(..., 'rb')
            t, t_dt64,freqsPower, Pdb_array, pp, data_dir, station = pickle.load(f)

#Loops through each day of data
for day in day_range:
    
    #define day of year as an integer
    dt=datetime.datetime(year,month,day, hr_s,min_s,sec_s)
    dayofyear=int(dt.strftime('%j'))
    
    
    #find frequencies index that represents the min and max frequencies
    freq1=np.where((freqsPower>Freqmin))[0][0]
    freq10=np.where((freqsPower>Freqmax))[0][0]
    
    # Looking at power at a daily time scale; defining the start and ending index of a day in the power dataset
    starting_index=np.where(t==UTCDateTime(year,month,day,hr_s,min_s))[0][0]
    try:
        ending_index= np.where(t==UTCDateTime(year,month,day+1,hr_s,min_s))[0][0] 
    except ValueError:
        ending_index= np.where(t==UTCDateTime(year,month+1,1,hr_s,min_s))[0][0]
     
    # remove the times that fall between the starting and ending index
    time=t[starting_index:ending_index]
    
    #Iitiate power array
    Power=np.zeros((len(time),len(freqsPower)))
    
    # Remove the power for the one day in the data set and find the median power value for each frequecny within the freq range
    for x in np.arange(starting_index,ending_index,1):
        Power[x-starting_index,:]=Pdb_array[:,x]
    Power_average=np.median(Power, axis=0)[freq1:freq10]
    
    #Smooth the power and remove the end points which are artifacts of the smoothing
    Power_average=np.delete(smooth(Power_average,smoothing_number),0)[:-1]

    #Find the freqeuencies where local power mininmum and maximums occur
    locmin_index=relex(Power_average,np.less)[0]
    locmax_index=relex(Power_average,np.greater)[0]
   
    #Check that we are starting with a local minimum value   
    if locmax_index[0] < locmin_index[0]:
       #if we start with a local max, assign the first point in the data set to be a minimum  
       locmin_index= np.insert(locmin_index,0,0)

    #Find the power values associated with the local minimums and maximums
    locmin_value=np.take(Power_average,locmin_index)
    locmax_value=np.take(Power_average,locmax_index)
     
    #find the vertical distance between all local minimums and maximums  
    for n in np.arange(0,len(locmin_value)-1,1):
        Distance.append(locmax_value[n]-locmin_value[n])
        Distance.append(locmax_value[n]-locmin_value[n+1])

# Finds the small distance that is the threshold for unnessecary local minimum values        
LL_constraint=np.percentile(Distance,20)    # 20th percentile can be changed by user


# Finds the distance that will be the power threshold for defining power peaks 
Power_constraint= np.percentile(Distance,50)    #50th percentile can be changed by user



#----------------------------------------------------------------------------




















