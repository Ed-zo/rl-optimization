# -*- coding: utf-8 -*-
"""
Created on Mon May  2 06:28:01 2016

@author: majer
"""

import time
import json
import os.path
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
        print str(t[1]) + "\t" + str(t[2]) + "\t" + str(t[3]) + "\t" + strTime(t[4]) + "\t" + str(t[5]) + "\t" + strTime(t[6]) + "\t" + str(t[9]) + "\t" + str(t[10]) + "\t" + str(t[11])

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
        prejazd_pred = 0
        prejazd_po = 0
        statie_po = 0
        trip = [ id, line, trip, depstop, deptime, arrstop, arrtime, prev, succ, prejazd_pred, prejazd_po, statie_po ]
        trips[id] = trip
    f.close()
    return trips

def createModel( trips, distances ):
    m = Model()
    x = {}
    cx = {}
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
                cx[i,j] = dist

    u = {}
    cu = {}
    for j in trips.keys():
        trip_j = trips[j]
        dep_stop_j = trip_j[3]
        dep_time_j = trip_j[4]
        dist = distances[depo,dep_stop_j]
        u[j] = m.addVar(vtype=GRB.BINARY, name='u'+str(j))
        cu[j] = dist

    v = {}
    cv = {}
    for i in trips.keys():
        trip_i = trips[i]
        arr_stop_i = trip_i[5]
        arr_time_i = trip_i[6]
        dist = distances[arr_stop_i,depo]
        v[i] = m.addVar(vtype=GRB.BINARY, name='v'+str(i))
        cv[i] = dist

    m.update()

    expr = LinExpr()
    expr += quicksum(cx[i,j] * x[i,j] for (i, j) in x.keys())
    expr += quicksum(cu[j] * u[j] for j in u.keys())
    expr += quicksum(cv[i] * v[i] for i in v.keys())
    m.setObjective(expr, GRB.MINIMIZE)

    m.update()

    for k in trips.keys():
        expr = LinExpr()
        expr += u[k]
        expr += quicksum( x[i,j] for (i, j) in x.keys() if k == j)
        m.addConstr( expr, GRB.EQUAL, 1 )

    for k in trips.keys():
        expr = LinExpr()
        expr += v[k]
        expr += quicksum( x[i,j] for (i, j) in x.keys() if k == i)
        m.addConstr( expr, GRB.EQUAL, 1 )

    m.addConstr( quicksum(x[i,j] for (i, j) in x.keys()), GRB.EQUAL, len(trips) - 39 )

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

    rotations = []
    for i in heads:
        rot = []
        while i != None:
            rot.append( trips[i] )
            i = trips[i][8]

        rotations.append(rot)

    return rotations

def printTurnusy( rotations, distances ):

    print "Tur\tZac\tKon\tPrist\tOdst\tPrej"

    total_pristav = 0;
    total_odstav = 0;
    total_prejazd = 0;
    for k in range(0, len(rotations)):

        rot = rotations[k]
        pristav = distances[depo, rot[0][3]]
        odstav = distances[rot[-1][5], depo]

        prejazd = 0
        for i in range(0, len(rot) - 1):
            u = rot[i][5]
            v = rot[i+1][3]
            prejazd = prejazd + distances[u, v]

        zaciatok = rot[0][4] - pristav
        koniec = rot[-1][6] + odstav

        print str(k+1) + "\t" + strTime( zaciatok ) + "\t" + strTime( koniec ) + "\t" + str( pristav ) + "\t" + str( odstav ) + "\t" + str( prejazd )

        total_pristav += pristav
        total_odstav += odstav
        total_prejazd += prejazd

    print "\t\t\t" + str( total_pristav ) + "\t" + str( total_odstav ) + "\t" + str( total_prejazd )
    print


def spocitajJazdu( jazda, z, k ):

    spolu = 0
    for j in jazda:

        if j[1] < z or j[0] > k:
            continue

        spolu += min(k, j[1]) - max(z, j[0])

    return spolu

def dajSpojeBP( rotation, zac, kon ):

    spoje = []
    for s in rotation:
        z = s[4] - s[9]
        k = s[6] + s[10]
        if k > zac and z < kon:
            spoje.append(s)

    return spoje

def printTurnus( tur, rotation, distances ):

    pristav = distances[depo, rotation[0][3]]
    rotation[0][9] = pristav

    odstav = distances[rotation[-1][5], depo]
    rotation[-1][10] = odstav

    zaciatok = rotation[0][4] - pristav
    koniec = rotation[-1][6] + odstav

    prejazd_spolu = 0
    for i in range(0, len(rotation) - 1):
        u = rotation[i][5]
        v = rotation[i+1][3]
        prejazd = distances[u, v]
        rotation[i][10] = prejazd
        rotation[i][11] = rotation[i+1][4] - rotation[i][6] - prejazd
        prejazd_spolu += prejazd

    print "Turnus: ", tur+1
    for t in rotation:
        print str(t[1]) + "\t" + str(t[2]) + "\t" + str(t[3]) + "\t" + strTime(t[4]) + "\t" + str(t[5]) + "\t" + strTime(t[6]) + "\t" + str(t[9]) + "\t" + str(t[10]) + "\t" + str(t[11])

    print

def kontrolujTurnus( tur, rotation, distances ):

    spoje = []

    pristav = distances[depo, rotation[0][3]]
    rotation[0][9] = pristav

    odstav = distances[rotation[-1][5], depo]
    rotation[-1][10] = odstav

    zaciatok = rotation[0][4] - pristav
    koniec = rotation[-1][6] + odstav

    prejazd_spolu = 0
    for i in range(0, len(rotation) - 1):
        u = rotation[i][5]
        v = rotation[i+1][3]
        prejazd = distances[u, v]
        rotation[i][10] = prejazd
        rotation[i][11] = rotation[i+1][4] - rotation[i][6] - prejazd
        prejazd_spolu += prejazd

    zaciatok = rotation[0][4] - rotation[0][9]
    koniec = rotation[-1][6] + rotation[-1][10]

    i = 0
    jazda = []
    while i < len(rotation):
        z = rotation[i][4] - rotation[i][9]
        while True:
            k = rotation[i][6] + rotation[i][10]
            if i == len(rotation) -  1:
                break
            p = (rotation[i+1][4] - rotation[i+1][9]) - k
            if p >= 10:
                break
            i += 1
        jazda.append((z,k))
        i += 1

    for i in range(0, len(jazda)):
        z = jazda[i][0]
        k = z + 270
        jaz = spocitajJazdu( jazda, z, k )
        if jaz > 240:
            print "Porusenie BP v turnuse ", tur+1, " v case od ", strTime(z), " do ", strTime(k), ", jazda = ", jaz, "!"
            sp = dajSpojeBP( rotation, z, k )
            if len(spoje) == 0 or spoje[-1] != sp:
                spoje.append(sp)

        k = jazda[i][1]
        z = k - 270
        jaz = spocitajJazdu( jazda, z, k )
        if jaz > 240:
            print "Porusenie BP v turnuse ", tur+1, " v case od ", strTime(z), " do ", strTime(k), ", jazda = ", jaz, "!"
            sp = dajSpojeBP( rotation, z, k )
            if len(spoje) == 0 or spoje[-1] != sp:
                spoje.append(sp)

    return spoje

distances = loadDistances("dist.txt")
trips = loadTrips("trips.txt")
print "Trips: ", len(trips)

startTime = time.clock()
model, x = createModel(trips, distances)
finishTime = time.clock()
print "Model vytvoreny za ", finishTime - startTime, " sekund."

startTime = finishTime
model.optimize()

iteracia = 1
while True:

    finishTime = time.clock()
    print "Iteracia ", iteracia, ", model vypocitany za ", finishTime - startTime, " sekund."

    startTime = finishTime
    rotations = getResult(trips,distances,model,x)

    spoje = []
    for k in range(0, len(rotations)):
        for ss in kontrolujTurnus( k, rotations[k], distances ):
            spoje.append(ss)

    if len(spoje) == 0:
        print "Iteracia: ", iteracia, ", HOTOVO "
        break
    else:
        print "Iteracia: ", iteracia, ", pocet zistenych poruseni BP: ", len(spoje)
        print "Porusenia BP:"
        for ss in spoje:
            for s in ss:
                printTrip(s)
            print
        
        for ss in spoje:
            expr = LinExpr();
            for i in range(0,len(ss)-1):
                expr += x[ss[i][0],ss[i+1][0]]
            model.addConstr(expr, GRB.LESS_EQUAL, len(ss) - 2)

        iteracia += 1
        model.update()

        finishTime = time.clock()
        print "Iteracia ", iteracia, ", model upraveny za ", finishTime - startTime, " sekund."

        startTime = finishTime
        model.optimize()
    
printTurnusy(rotations, distances)

for k in range(0, len(rotations)):
    printTurnus( k, rotations[k], distances )
print

