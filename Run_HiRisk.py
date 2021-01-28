from optimization_All import *

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
dictVar['hiRiskCtrl'] = [0.66,0.8]

## OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[0],] 

## RUN SIMULATION
optimalMinCost_Budget_Simulation(dictVar,t1 = 51, tstep = 2, minCost = 0, maxCost =  5.1E7, cStep = 2.5E6)

## 2D PLOT WITH DEATHS AS WHITE CONTOURS, COST AS COLOR REGIONS
# Using constant cost strategy

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# optimalMinCost_Budget_plots(dictVar)

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# dictVar['filename'] = [ files[1] , 'optimalMinCost_Budget_Basic_Pareto.json' ] 
# dictVar['recomputeData'] = True # SET FALSE IF YOU HAVE ALREADY COMPUTED THE DATA AND SAVED THE FILE.
# ctrlMeasures_paretoHeatmaps_Budget(dictVar)



###############################
### R-Frac Strategy 
###############################  
dictVar['hiRiskCtrl'] = [0.66,0.8]
dictVar['rFrac'] = True
 
## OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[1],] 

## RUN SIMULATION
optimalMinCost_R_Simulation(dictVar,t1 = 51, tstep = 2, f0 = 0.5, ff = 1.0, fstep =0.02)

## 2D PLOT WITH DEATHS AS WHITE CONTOURS, COST AS COLOR REGIONS
# optimalMinCost_R_plots(dictVar)

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# dictVar['filename'] = [ files[1] , 'optimalMinCost_rFrac_Basic_Pareto.json' ]
# dictVar['recomputeData'] = True # SET FALSE IF YOU HAVE ALREADY COMPUTED THE DATA AND SAVED THE FILE.
# ctrlMeasures_paretoHeatmaps_R(dictVar)



###############################
### R-Target Strategy 
############################### 
dictVar['hiRiskCtrl'] = [0.66,0.8]
dictVar['rFrac'] = False 

## OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[2],]

## RUN SIMULATION
optimalMinCost_R_Simulation(dictVar,t1 = 51, tstep = 2, f0 = 0.5, ff = 1.0, fstep =0.02)

## 2D PLOT WITH DEATHS AS WHITE CONTOURS, COST AS COLOR REGIONS
# optimalMinCost_R_plots(dictVar)

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# dictVar['filename'] = [ files[1] , 'optimalMinCost_rTgt_Basic_Pareto.json' ]
# dictVar['recomputeData'] = True # SET FALSE IF YOU HAVE ALREADY COMPUTED THE DATA AND SAVED THE FILE.
# ctrlMeasures_paretoHeatmaps_R(dictVar)

