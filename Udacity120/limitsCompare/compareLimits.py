#!/usr/bin/python


import numpy

oldLimits = dict()
newLimits = dict()
changedLimits = list()
missingLimits = list()
    
limitsFileHandler = open("/downloads/TradingLimitInfos-81252.txt", "r")
for line in limitsFileHandler:
        limits = line.split("\t");
        if (limits[0] == 'Y' and not limits[2] == "OrderPriceAway" and "Roll" not in limits[6] and "hzou" not in limits[6]):
                key = limits[6] + '_' + limits[2]
                oldLimits[key] = limits[10]
limitsFileHandler.close()
                
limitsFileHandler = open("/downloads/TradingLimitInfos.txt", "r")
for line in limitsFileHandler:
        limits = line.split("\t");
        if (limits[0] == 'Y' and not limits[2] == "OrderPriceAway" and "Roll" not in limits[6]):
                key = limits[6] + '_' + limits[2]
                newLimits[key] = limits[10]
limitsFileHandler.close()
                
for key in oldLimits:
    if (key in newLimits):
        if (oldLimits[key] != newLimits[key]):
            changedLimits.append(">> " + key + ": " + oldLimits[key] + " <> " + newLimits[key])
    else:
        missingLimits.append("-- " + key + ": Missing");
    
changedLimits.sort()
missingLimits.sort()

for key in changedLimits:
            print(key)
    
for key in missingLimits:
        print(key)
                
    
