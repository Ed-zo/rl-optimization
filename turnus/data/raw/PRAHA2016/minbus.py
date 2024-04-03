# -*- coding: utf-8 -*-
"""
Created on Mon May  2 06:28:01 2016

@author: majer
"""

from gurobipy import *

depo = 59

def loadDistances( fileName ):
    distances = {}
    f = open( fileName, "r" )
    for l in f.readlines():
        cols = l.split()
        u = int(cols[0])
        v = int(cols[1])
        c = int(cols[2])
        distances[u,v] = c
        distances[v,u] = c
        distances[u,u] = 0
        distances[v,v] = 0
    f.close()
    return distances
    
def parseTime( time ):
    cols = time.split(":")
    hour = int(cols[0])
    minute = int(cols[1])
    return 60 * hour + minute
    
def strTime( time ):
    h = time / 60    
    m = time % 60
    return "{0:02d}:{1:02d}".format(h, m)
    
def printTrip( t ):
    print str(t[1]) + "\t" + str(t[2]) + "\t" + str(t[3]) + "\t" + strTime(t[4]) + "\t" + str(t[5]) + "\t" + strTime(t[6])
    
def loadTrips( fileName ):
    trips = {}
    f = open( fileName, "r" )
    for l in f.readlines():
        cols = l.split()
        line = int(cols[0])
        trip = int(cols[1])
        id = 10000 * line + trip;
        depstop = int(cols[2])
        deptime = parseTime(cols[3])
        arrstop = int(cols[4])
        arrtime = parseTime(cols[5])
        prev = None
        succ = None
        trip = [ id, line, trip, depstop, deptime, arrstop, arrtime, prev, succ ]
        trips[id] = trip
    f.close()
    return trips

def createModel( trips, distances ):
    m = Model()
    x = {}
    for i in trips.keys():
        for j in trips.keys():
            trip_i = trips[i]
            arr_stop_i = trip_i[5]
            arr_time_i = trip_i[6]
            trip_j = trips[j]
            dep_stop_j = trip_j[3]
            dep_time_j = trip_j[4]
            dist = distances[arr_stop_i,dep_stop_j]
            if arr_time_i + dist <= dep_time_j:
                x[i,j] = m.addVar(vtype=GRB.BINARY, name='x'+str(i)+'_'+str(j))
        
    m.update()
        
    obj = quicksum(x[i,j] for (i, j) in x.keys())
    m.setObjective(obj, GRB.MAXIMIZE)
        
    m.update()
    
    for k in trips.keys():
        m.addConstr(quicksum( x[i,j] for (i, j) in x.keys() if k == j) <= 1 )
    
    for k in trips.keys():
        m.addConstr(quicksum( x[i,j] for (i, j) in x.keys() if k == i) <= 1 )
    
    m.update()
                
    return m, x

def getResult( trips, distances, m, x ):
    
    for i in trips.keys():
        trips[i][7] = None
        trips[i][8] = None
        
    for (i, j) in x.keys():
        if x[i,j].X == 1:
            trips[i][8] = j
            trips[j][7] = i

    heads = []
    for i in trips.keys():
        if trips[i][7] == None:
            heads.append(i)
            
    return heads
    
def printTurnusy( trips, distances, heads ):

    print "Tur\tZac\tKon\tPrist\tOdst\tPrej"

    n = 1
    total_prej = 0;
    for head in heads:

        pristav = distances[depo, trips[head][3]]
        zaciatok = trips[head][4] - pristav

        # najdem posledny spoj
        tail = head
        while trips[tail][8] != None:
            tail = trips[tail][8]
            
        odstav = distances[depo, trips[tail][5]]
        koniec = trips[tail][6] + odstav
        
        i = head
        prejazd = 0
        while trips[i][8] != None:
            u = trips[i][5]
            i = trips[i][8]
            v = trips[i][3]
            prejazd = prejazd + distances[u, v]
        
        print str(n) + "\t" + strTime( zaciatok ) + "\t" + strTime( koniec ) + "\t" + str( pristav ) + "\t" + str( odstav ) + "\t" + str( prejazd ) 
        
        total_prej = total_prej + pristav + odstav + prejazd
        n = n + 1
    
    print "Prejazdy spolu: ", total_prej
    print

distances = loadDistances("dist.txt")
trips = loadTrips("trips.txt")
print "Trips: ", len(trips)

model, x = createModel(trips, distances)
model.optimize()

boards = len(trips) - model.ObjVal
print "Boards: ", boards

heads = getResult(trips,distances,model,x)
printTurnusy(trips,distances,heads)

for k in heads:
    i = k
    while i != None:
        printTrip(trips[i])
        i = trips[i][8]
    print
print
