from optimization_All import *
import numpy as np

###############
### ADDITIONAL CONFIGURATIONS CAN BE FOUND IN THE 'initilization()' METHOD IN 'optimization_all'
###############
dictVar = initialization()

files = ['optimalMinCost_Budget_HiRiskMax.json',\
        'optimalMinCost_rFrac_HiRiskMax.json',\
        'optimalMinCost_rTgt_HiRiskMax.json']


###############################
### Constant Cost Strategy 
###############################  
dictVar['hiRiskCtrl'] = True

## OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[0],] 

## RUN SIMULATION
# optimalMinCost_Budget_Simulation(dictVar,t1 = 51, tstep = 2, minCost = 0, maxCost =  5.1E7, cStep = 2.5E6)
# optimalMinCost_Budget_Simulation(dictVar,t1 = 51, tstep = 4, minCost = 0, maxCost =  5.1E7, cStep = 5E6)

## 2D PLOT WITH DEATHS AS WHITE CONTOURS, COST AS COLOR REGIONS
# optimalMinCost_Budget_plots(dictVar)

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# dictVar['filename'] = [ files[0] , 'optimalMinCost_Budget_HiRisk_Pareto.json' ] 
# dictVar['recomputeData'] = False # SET FALSE IF YOU HAVE ALREADY COMPUTED THE DATA AND SAVED THE FILE.
# ctrlMeasures_paretoHeatmaps_Budget(dictVar)



###############################
### R-Frac Strategy 
###############################  
dictVar['hiRiskCtrl'] = True
dictVar['rFrac'] = True
 
# OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[1],] 

## RUN SIMULATION
# optimalMinCost_R_Simulation(dictVar,t1 = 51, tstep = 2, f0 = 0.5, ff = 1.0, fstep =0.02)
# optimalMinCost_R_Simulation(dictVar,t1 = 51, tstep = 5, f0 = 0.5, ff = 1.0, fstep =0.1)

## 2D PLOT WITH DEATHS AS WHITE CONTOURS, COST AS COLOR REGION
# optimalMinCost_R_plots(dictVar)

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# dictVar['filename'] = [ files[1] , 'optimalMinCost_rFrac_HiRisk_Pareto.json' ]
# dictVar['recomputeData'] = False # SET FALSE IF YOU HAVE ALREADY COMPUTED THE DATA AND SAVED THE FILE.
# ctrlMeasures_paretoHeatmaps_R(dictVar)


###############################
### R-Target Strategy 
############################### 
dictVar['hiRiskCtrl'] = True
dictVar['rFrac'] = False 

## OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[2],]

## RUN SIMULATION
# optimalMinCost_R_Simulation(dictVar,t1 = 51, tstep = 2, f0 = 0.5, ff = 1.0, fstep =0.02)
# optimalMinCost_R_Simulation(dictVar,t1 = 51, tstep = 5, f0 = 0.5, ff = 1.0, fstep =0.1)

## 2D PLOT WITH DEATHS AS WHITE CONTOURS, COST AS COLOR REGIONS
# optimalMinCost_R_plots(dictVar)

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# dictVar['filename'] = [ files[2] , 'optimalMinCost_rTgt_HiRisk_Pareto.json' ]
# dictVar['recomputeData'] = False # SET FALSE IF YOU HAVE ALREADY COMPUTED THE DATA AND SAVED THE FILE.
# ctrlMeasures_paretoHeatmaps_R(dictVar)

