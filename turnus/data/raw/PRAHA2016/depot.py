# -*- coding: utf-8 -*-
"""
Created on Mon May  2 06:28:01 2016

@author: majer
"""

from gurobipy import *

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
    print t[1], t[2], t[3], strTime(t[4]), t[5], strTime(t[6])
    
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

def createDepots( trips, distances, depot ):
    depots = {}
#    for i in range(600, 720):
#        line = 999
#        trip = 1 + len(depots)
#        id = 10000 * line + trip;
#        depstop = depot
#        deptime = i
#        arrstop = depot
#        arrtime = i + 60
#        prev = None
#        succ = None
#        trip = [ id, line, trip, depstop, deptime, arrstop, arrtime, prev, succ ]
#        depots[id] = trip
#        printTrip(trip)
        
    for i in trips.keys():
        trip_i = trips[i]
        arr_stop_i = trip_i[3]
        arr_time_i = trip_i[4]
        if arr_time_i >= 480 and arr_time_i < 660:
            line = 999
            trip = 1 + len(depots)
            id = 10000 * line + trip;
            depstop = depot
            deptime = arr_time_i + distances[arr_stop_i,depot]
            arrstop = depot
            arrtime = deptime + 60
            prev = None
            succ = None
            trip = [ id, line, trip, depstop, deptime, arrstop, arrtime, prev, succ ]
            depots[id] = trip
            printTrip(trip)
            
    return depots

def createModel1( trips, distances ):
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

def createModel2( trips, depots, distances, nBoards ):
    m = Model()
    x = {}
    z = {}
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

    for i in trips.keys():
        for j in depots.keys():
            trip_i = trips[i]
            trip_j = depots[j]

            arr_stop_i = trip_i[5]
            arr_time_i = trip_i[6]
            dep_stop_j = trip_j[3]
            dep_time_j = trip_j[4]
            dist = distances[arr_stop_i,dep_stop_j]
            if arr_time_i + dist <= dep_time_j:
                x[i,j] = m.addVar(vtype=GRB.BINARY, name='x'+str(i)+'_'+str(j))

            dep_stop_i = trip_i[3]
            dep_time_i = trip_i[4]
            arr_stop_j = trip_j[5]
            arr_time_j = trip_j[6]
            dist = distances[arr_stop_j,dep_stop_i]
            if arr_time_j + dist <= dep_time_i:
                x[j,i] = m.addVar(vtype=GRB.BINARY, name='x'+str(j)+'_'+str(i))

    for i in depots.keys():
        z[i] = m.addVar(vtype=GRB.BINARY, name='z'+str(i))
        
    m.update()
        
    obj = quicksum(x[i,j] for (i, j) in x.keys())
    m.setObjective(obj, GRB.MAXIMIZE)
        
    m.update()
    
    for k in trips.keys():
        m.addConstr(quicksum( x[i,j] for (i, j) in x.keys() if k == j) <= 1 )
    
    for k in depots.keys():
        m.addConstr(z[k] + quicksum( x[i,j] for (i, j) in x.keys() if k == j) <= 1 )
    
    for k in trips.keys():
        m.addConstr(quicksum( x[i,j] for (i, j) in x.keys() if k == i) <= 1 )
    
    for k in depots.keys():
        m.addConstr(z[k] + quicksum( x[i,j] for (i, j) in x.keys() if k == i) <= 1 )

    m.addConstr(len(depots.keys()) - quicksum( z[i] for i in z.keys()) == nBoards )
    
    m.update()
                
    return m, x, z

distances = loadDistances("dist.txt")
trips = loadTrips("trips.txt")
print "Trips: ", len(trips)

model, x = createModel1(trips, distances)
model.optimize()

boards = len(trips) - model.ObjVal
print "Boards: ", boards

depots = createDepots(trips, distances, 59)
print "Depots: ", len(depots)
    
alltrips = {}
for i in trips.keys():
    alltrips[i] = trips[i]
for i in depots.keys():
    alltrips[i] = depots[i]

finished = False

while not finished:
    model, x, z = createModel2(trips, depots, distances, boards)
    model.optimize()

    for i in alltrips.keys():
        alltrips[i][7] = None
        alltrips[i][8] = None
        
    
    for (i, j) in x.keys():
        if x[i,j].X == 1:
            alltrips[i][8] = j
            alltrips[j][7] = i
            
    heads = []
    for i in alltrips.keys():
        if alltrips[i][7] == None and ( alltrips[i][1] != 999 or z[i].X == 0 ):
            heads.append(i)
    
    print "Nr of boards: ", len(heads)
    
    brds = 0
    for k in range(0, len(heads)):
        deps = 0
        i = heads[k]
        while i != None:
            if alltrips[i][1] == 999:
                deps = deps + 1
            i = alltrips[i][8]
        if deps == 0:
            brds = brds + 1
    print "Nr of boards without depot: ", brds
    print    
    
    if brds == 0:
        finished = True
    else:
        boards = boards + 1


print "Boards: "
for k in heads:
    i = k
    while i != None:
        printTrip(alltrips[i])
        i = alltrips[i][8]
    print
print
