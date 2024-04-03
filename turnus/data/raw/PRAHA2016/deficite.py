# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:53:19 2016

@author: majer
"""
    
def parseTime( time ):
    cols = time.split(":")
    hour = int(cols[0])
    minute = int(cols[1])
    return 60 * hour + minute
    
def strTime( time ):
    h = time / 60    
    m = time % 60
    return "{0:02d}:{1:02d}".format(h, m)

values = [ 0 for i in range(0,1440)]

f = open( "trips.txt", "r" )
for l in f.readlines():
    cols = l.split()
    line = int(cols[0])
    trip = int(cols[1])
    id = 10000 * line + trip;
    depstop = int(cols[2])
    deptime = parseTime(cols[3])
    arrstop = int(cols[4])
    arrtime = parseTime(cols[5])
    for i in range(deptime, arrtime):
        values[i] += 1
f.close()

for i in range(0, 1440):
    print strTime(i) + ":00", values[i]
    print strTime(i) + ":59", values[i]
