from optimization_All import *

###############
### ADDITIONAL CONFIGURATIONS CAN BE FOUND IN THE 'initilization()' METHOD IN 'optimization_all'
###############
dictVar = initialization()

files = ['optimalMinCost_Budget_Basic.json',\
        'optimalMinCost_rFrac_Basic.json',\
        'optimalMinCost_rTgt_Basic.json']


###############################
### Constant Cost Strategy 
###############################  

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

## PARETO PLOTS OF COST VERSUS DEATHS ( START DATA NOT INCLUDED)
# dictVar['filename'] = files[0]
# pareto_plot(dictVar)



###############################
### R-Frac Strategy 
###############################  

## OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[1],]
dictVar['rFrac'] = True

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

## OUTPUT FILE FOR SIMULATION / INPUT FILE FOR PLOTS
dictVar['filename'] = [files[2],]

dictVar['rFrac'] = False
optimalMinCost_R_Simulation(dictVar,t1 = 51, tstep = 2, f0 = 0.5, ff = 1.0, fstep =0.02)

## 2D PLOT WITH DEATHS AS WHITE CONTOURS, COST AS COLOR REGIONS
# optimalMinCost_R_plots(dictVar)

## 2D PLOTS SHOWING DAY BY DAY INFORMATION BASED ON THE OPTIMAL VALUES ON EACH CONTOUR
# dictVar['filename'] = [ files[1] , 'optimalMinCost_rTgt_Basic_Pareto.json' ]
# dictVar['recomputeData'] = True # SET FALSE IF YOU HAVE ALREADY COMPUTED THE DATA AND SAVED THE FILE.
# ctrlMeasures_paretoHeatmaps_R(dictVar)






























