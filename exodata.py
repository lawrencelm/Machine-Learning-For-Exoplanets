#! /usr/bin/env python

import numpy as np

#1. Load the data into python
# There are two datasets in use here exodata.csv which contains potentially habitable planets
# and exodata1.csv which contains non-classified planets
habitable_file = 'exoplanets/exodata.csv'
nonhabitable_file = 'exoplanets/exodata1.csv'
names = "radius, flux, temp, period"
habitable = np.genfromtxt(habitable_file, delimiter=',',\
    missing_values='',filling_values=np.nan, usecols=(1,2,3,4), names=names)
nonhabitable = np.genfromtxt(nonhabitable_file, delimiter=',',\
    missing_values='',filling_values=np.nan, usecols=(2,3,4,5), names=names)

selection1 = np.logical_or(np.isnan(nonhabitable['radius']),np.isnan(nonhabitable['flux']))
selection2 = np.logical_or(np.isnan(nonhabitable['temp']), np.isnan(nonhabitable['period']))
sel = np.logical_not(np.logical_or(selection1, selection2))

#2. Convert all the data into binary values by comparing them against the earth values
habitable['radius'] = habitable['radius'] > 1.0
nonhabitable['radius'] = nonhabitable['radius'] > 1.0
habitable['flux'] = habitable['flux'] > 1.0
nonhabitable['flux'] = nonhabitable['flux'] > 1.0
habitable['temp'] = habitable['temp'] > 288
nonhabitable['temp'] = nonhabitable['temp'] > 288
habitable['period'] = habitable['period'] > 1.0
nonhabitable['period'] = nonhabitable['period'] > 1.0

#Now we can cast the new boolean values into 1 for True and 0 for False. The names allow us to
#directly access the columns we want in the structured array.
habitable = habitable.astype([('radius', int), ('flux', int), ('temp', int), ('period', int)])
nonhabitable = nonhabitable.astype([('radius', int), ('flux', int), ('temp', int), ('period', int)])

#3. Make the training data set by combining half the data from habitable_file and 200 random non_habitable 
# exoplanets from the nonhabitable dataset.
# All variables are binary. The esi is the measure of similarity of a planet to earth. All planets in 
# the habitable data set have an esi of 1. 

num_habitable = len(habitable['radius'])/2
num_nonhabitable = 200

# Combine all the data into one train data array
train_y = np.array([1]*num_habitable + [0]* 200) #esi values
train_data = np.hstack((habitable[:num_habitable], nonhabitable[:200]))

#4. Lets create the test data array
test_y = np.array([1]*(len(habitable) - num_habitable) + [0]*(len(nonhabitable)-200))
test_data = np.hstack((habitable[num_habitable:], nonhabitable[200:]))

#5. Now we need to write all the data into txt files
