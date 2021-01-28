# Most routines with 3 files need the order as: budget, fraction, target

from optimization_All import *

###############
### ADDITIONAL CONFIGURATIONS CAN BE FOUND IN THE 'initilization()' METHOD IN 'optimization_all'
###############
dictVar = initialization()

# THESE FILES ARE ACQUIRED AFTER RUNNING 
# THE SIMULATIONS FROM 'Run_Basic.py' AND 'Run_HiRIsk'

filename = ['optimalMinCost_Budget_Basic.json',\
            'optimalMinCost_rFrac_Basic.json',\
            'optimalMinCost_rTgt_Basic.json',\
            'optimalMinCost_Budget_HiRiskMax.json',\
            'optimalMinCost_rFrac_HiRiskMax.json',\
            'optimalMinCost_rTgt_HiRiskMax.json']


dictVar['filename'] = [ filename[0], filename[1], filename[2]]   # COMPARE REGULAR SIMULATIONS
# dictVar['filename'] = [ filename[3], filename[4], filename[5]] # COMPARE HI-RISK SIMULATIONS

###################
### Plots
###################

## PARETO PLOTS OF COST VERSUS DEATHS ( START DATE NOT INCLUDED)
pareto_plot(dictVar)


## LINE GRAPHS FOR DIFFERENT STRATEGY CLASSES AND
## FIXED DEATH LEVELS, PLOT COST AS A FUNCTION OF START DAY
implementCost_VS_StartDay(dictVar, deathValue = [ 40E3,80E3])


## LINE GRAPHS FOR DIFFERENT STRATEGY CLASSES AND
## FIXED COST LEVELS, PLOT DEATH AS A FUNCTION OF START DAY  
Deaths_VS_StartDay(dictVar,costValue = [ 2E9,4E9,6E9])















