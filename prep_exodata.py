#! /usr/bin/env python

import numpy as np

if __name__ =='__main__':
    filename = 'exoplanets/planets (2).csv'
    names = "period, semi-major, eccentricity, density, inclination, distance,temp, stellar mass, stellar radius, name, mass, radius, age"
    exodata = np.genfromtxt(filename, delimiter=',', skip_header=19,\
        missing_values='',filling_values=np.nan, usecols=(1,2,3,4,5,6,7,8,9,11,12,13,14), names=names)

    selection1 = np.logical_or(np.isnan(exodata['radius']),np.isnan(exodata['density']))
    sel = np.logical_not(np.logical_or(selection1, np.isnan(exodata['temp'])))

    #Calculate the similarity index
    #Reference Values
    n = 3 #No of planetary properties
    earth_temp = 288
    earth_density = 5.51
    temp_index = (1-abs((exodata['temp'][sel] - earth_temp)))**(5.58/3)
    print temp_index
    radius_index = (1-(exodata['radius'][sel]-1)/(exodata['radius'][sel]+1))**(0.57/3)
    density_index = (1-(exodata['density'][sel] - earth_density)/(exodata['density'][sel] + earth_density))**(1.07/3)

    esi = temp_index * radius_index * density_index
