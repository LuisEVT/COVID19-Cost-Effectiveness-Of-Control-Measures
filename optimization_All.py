# Gives optional fixed control to high risk group when control begins
# Also gives option for fixed R target or fraction
# Filename and file type (pickle/json) for read/write is now in dictionary
# See initialization() for all dictionary variables 


# IMPORTS
import numpy as np
import scipy.optimize as opt 
from skimage import measure
from scipy.interpolate import griddata


# IMPORTS TO SAVE DATA
import json
import pickle
import os.path 

# PLOTS IMPORTS
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.figure import figaspect
from textwrap import wrap
from cycler import cycler # cycles colors and linestyles

### LOCAL IMPORTS
from controlfnsenumv5 import forwardSolve


### INITIALIZATION
def initialization():
    
    ######################
    ### THESE SETTINGS ARE CHANGED DEPENDING ON THE SIMULATION
    ### PLEASE TAKE A LOOK AT 'Run_Basic.py' AND 'Run_HiRisk.py' FOR CLARIFICATION
    #######################

    # THIS CAN BE SET FOR HIGH-RISK TARGETING STRATEGIES
    hiRiskCtrl = [0] #[0.6666,0.8]

    # rFrac: 'TRUE' MEANS THAT R_e TARGET IS GIVEN AS A FRACTION
    rFrac = True

    # recomputeData: 'FALSE' MEANS THAT DATA IN THE PARETO HEATMAPS IS ALREADY COMPUTED AND NO COMPUTATION IS NEEDED
    recomputeData = False
    


    # TO STAY CONSISTENT, THESE VALUES ARE USED THROUGHOUT EACH PLOT
    ## Plot parameters
    totCostLevels = np.arange(0.0,8.1,0.5)*1E9 # VALUES FOR CONTOURS
    deathLevels = np.array([0.1,0.4,2,4,6,8,10,12,13,14])*1E4 # VALUES FOR CONTOURS
    nLevel = 10 # NUMBER OF COST CONTOURS ( USED FOR THE ADDITIONAL PLOTS)

    # ADDITIONAL PLOTS CAN BE FOUND IN 'optimalMinCost_R_plots' and 'optimalMinCost_Budget_plots'
    # WHEN PARAMETER SET TO TRUE.
    showAllPlots = False 



    
    # Decide which xLimits,yLimits are appropriate
    xLimits = []
    xLimits = [0,20] 
    # xLimits = [0,11]
    # xLimits = [0,8E9]
    
    yLimits = []
    yLimits = [20000,140000] # Deaths  
    # yLimits =  [1E9,8E9] # Cost

    

    ###############
    ### CONTROL PARAMETERS
    ###############

    # MAX. VALUE FOR CONTROL MEASURES
    # [1]: LOW-RISK TESTING
    # [2]: HIGH-RISK TESTING
    # [3]: LOW-RISK SOCIAL DISTANCING
    # [4]: HIGH-RISK SOCIAL DISTANCING
    uMax = np.array([0.6666, 0.6666, 0.8, 0.8]) 

    # Set high risk controls (if two values are set, then these vals are fixed)
    # Stop control when infected level goes below this
    minInfected = 10 
 
    ## Timing
    tf = 180            # Final time
    nStep = tf          # Number of intervals where control can take place:
                        ## In the current plot routines, it is assumed that nStep=tf
    
    ###############
    ### INITIAL CONDITIONS
    ###############
    N = [1340000,423000]
    NA = [1340000,423000]
    xI = np.zeros(18)
    xI[1] = 150
    xI[10] = 50
    xI[0] = N[0]-xI[1] # Susceptible low risk
    xI[9] = N[1]-xI[10] # Susceptible high risk
    
    ## System parameters
    beta =0.0640
    Phi = np.array([[10.56,2.77],
                    [9.4,2.63]]) ###Contact matrix
    
    gamA = 1/4####1/gamA~4
    gamY = 1/4###gamA=gamY##1/gamA~4
    gamH = 1/10.7
    eta = 0.1695
    tau = 0.57 # asymptomatic proportion
    sig = 1/2.9### exposed rate
    rhoA = 1/2.3###rhoA=rhoY##1/rhoY~2.3
    rhoY = 1/2.3###1/rhoY~2.3
    P = 0.44
    wY =1.0
    wA =0.66
    
    # Derived system parameters
    IFR = np.array([0.6440/100,6.440/100])
    YFR = np.array([IFR[0]/tau,IFR[1]/tau]) 
    YHR = np.array([4.879/100,48.79/100])
    HFR = np.array([YFR[0]/YHR[0],YFR[1]/YHR[1]])   
    wP= P/(1-P) /(tau*wY/rhoY + (1-tau)*wA/rhoA) \
        * ((1-tau)*wA/gamA \
        + tau*wY* np.array([YHR[0]/eta+(1-YHR[0])/gamY, \
                          YHR[1]/eta+(1-YHR[1])/gamY]))            
    wPY = wP*wY
    wPA = wP*wA     
    Pi = gamY*np.array([YHR[0]/(eta+(gamY-eta)*YHR[0]),\
                      YHR[1]/(eta+(gamY-eta)*YHR[1])])# Array with two values
    
    # Additional system parameters    
    mu = 1/8.1###1/mu~8.1
    theta = 3000 #2352 ventilators in Houston (https://www.click2houston.com/health/2020/04/10/texas-medical-center-data-shows-icu-ventilator-capacity-vs-usage-during-coronavirus-outbreak/)
    nu = gamH*np.array([HFR[0]/(mu+(gamH-mu)*HFR[0]),\
                      HFR[1]/(mu+(gamH-mu)*HFR[1])])# Array with two values    
        
    ## Costs
    a = np.array([[0,2.3,27],[0,2.3,27]]) # Testing costs
    b = np.array([[0,0,40],[0,0,40]]) # Distancing costs
    c = [100,100]     # Cost 5: Opportunity cost for sickness per day (low and high risk)
    d =  [500,750]   # Cost 6: Hospitalization cost per day  (low and high risk)
    e = [1,1]#[100000,75000] # Death
    f = [1,1]#[5000,5000] # Cost of remaining infected
    

    params = [wY,wA,wPY,wPA,beta,sig,tau,rhoA,rhoY,
        gamA,gamY,gamH,Pi,eta,nu,mu,theta,Phi]     
    
    ### Dictionary for functions that require it
    dictVar={'N':N,
        'NA':NA,
        'xI':xI,
        'beta':beta,
        'Phi':Phi,
        'gamA':gamA,
        'gamY':gamY,
        'gamH':gamH,
        'eta':eta,
        'tau':tau,
        'sig':sig,
        'rhoA':rhoA,
        'rhoY':rhoY,
        'P':P,
        'wY':wY,
        'wA':wA,
        'IFR':IFR,
        'YFR':YFR,
        'YHR':YHR,
        'HFR':HFR,
        'wP':wP,
        'wPY':wPY,
        'wPA':wPA,
        'Pi':Pi,
        'mu':mu,
        'theta':theta,
        'nu':nu,
        'a':a,
        'b':b,
        'c':c,
        'd':d,
        'e':e,
        'f':f,
        'tf':tf,
        'nStep':nStep,
        'uMax':uMax,
        'params':params,
        'hiRiskCtrl':hiRiskCtrl, 
        'minInfected':minInfected,
        'rFrac':rFrac,
        'filename':'NO FILE SET',
        'totCostLevels':totCostLevels,
        'deathLevels':deathLevels,
        'nLevel':nLevel,
        'showAllPlots':showAllPlots,
        'xLimits':xLimits,
        'yLimits':yLimits,
        'recomputeData':recomputeData
        } 

    return dictVar


def compCost(u,dictVar):
    '''
    Parameters:
        dictVar: Dictionary with variables from initialization() 
        u: Control rates
            U[0]: Low risk testing rate ( 0-1 )
            U[1]: High risk resting rate ( 0-1 )
            U[2]: Low risk distance rate ( 0-1 )
            U[3]: High risk distance rate ( 0-1 )

    Purpose:
        COMPUTE THE MARGINAL COST FOR TESTING AND SOCIAL DISTANCING
    '''

    # INITIAL CONDITION
    
    N = dictVar['N']
    NA = dictVar['NA']
    a = dictVar['a']
    b = dictVar['b']


    # @@@@ Hard-wire equal levels 
    #u[0] = u[1]
    #u[2] = u[3]

    #######################
    ### COMPUTE TOTAL COST
    #######################

    # SOCIAL DISTANCE (LOW RISK)  
    distCost = b[0,0]*N[0]+b[0,1]*N[0]*u[2]+b[0,2]*N[0]*u[2]**2
    # SOCIAL DISTANCE (HIGH RISK)
    distCost = distCost + b[1,0]*N[1]+b[1,1]*N[1]*u[3]+b[1,2]*N[1]*u[3]**2

    # TESTING (LOW RISK) 
    testCost = (u[0]>0)*a[0,0]+a[0,1]*u[0]*NA[0]*u[0]+a[0,2]*NA[0]*u[0]**2
    # TESTING (HIGH RISK) 
    testCost = testCost + (u[1]>0)*a[1,0]+a[1,1]*u[1]*NA[1]*u[1]+a[1,2]*NA[1]*u[1]**2

    # COST FOR SOCIAL DISTANCING AND TESTING 
    return distCost + testCost


def compReproductionNumber(u,dictVar,Imm):
    '''
    Parameters:
        u: Control rates
            U[0]: Low risk testing rate ( 0-1 )
            U[1]: High risk resting rate ( 0-1 )
            U[2]: Low risk distance rate ( 0-1 )
            U[3]: High risk distance rate ( 0-1 )
        dictVar: Dictionary with variables from initialization()
        Imm: Immunity rate ( 0-1 )    
        
    Purpose:
        COMPUTE THE REPRODUCTION NUMBER 'RHO'
    '''
    
    # INITIAL CONDITION
    N = dictVar['N']
    
    # SYSTEM PARAMETERS
    beta = dictVar['beta']
    Phi = dictVar['Phi']
    gamA = dictVar['gamA']
    gamY = dictVar['gamY']
    eta = dictVar['eta']
    tau = dictVar['tau']
    sig = dictVar['sig']
    rhoA = dictVar['rhoA']
    rhoY = dictVar['rhoY']
    wY = dictVar['wY']
    wA = dictVar['wA']
    wPY = dictVar['wPY']
    wPA = dictVar['wPA']
    Pi = dictVar['Pi']
    
    # Allows setting high risk controls at fixed value
    hiRiskCtrl = dictVar['hiRiskCtrl']

    # POPULATION THAT IS NOT IMMUNE 
    S = (1-Imm)*np.array(N)
    
    # (Optional) fix high risk controls
    if len(hiRiskCtrl)==2:
        u[1],u[3] = hiRiskCtrl

    ###############################
    ### COMPUTE REPRODUCTION NUMBER
    ###############################
    
    ## Create next generation matrix
    # First, compute F
    # Avoid division by 0
    SdivN = [[0,0],[0,0]]
    if N[0] > 0:
        SdivN[0][0] = S[0]/N[0]
        SdivN[1][0] = S[1]/N[0]
    if N[1] > 0:
        SdivN[0][1] = S[0]/N[1]
        SdivN[1][1] = S[1]/N[1]
    
    F00 = np.array([np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])*(1-u[2])*beta*SdivN[0][0]*Phi[0,0],
            [(1-tau)*sig,0,0,0,0],
            [tau*sig,0,0,0,0],
            [0,rhoA,0,0,0],
            [0,0,rhoY,0,0]])
    
    F11 = np.array([
            np.array([0,(1-u[1])*wPA[0],(1-u[1])*wPY[0],(1-u[1])*wA,wY])*(1-u[3])*beta*SdivN[1][1]*Phi[1,1],
            [(1-tau)*sig,0,0,0,0],
            [tau*sig,0,0,0,0],
            [0,rhoA,0,0,0],
            [0,0,rhoY,0,0]])
    
    F01 = np.array([
            np.array([0,(1-u[1])*wPA[1],(1-u[1])**wPY[1],(1-u[1])*wA,wY])*(1-u[3])*beta*SdivN[0][1]*Phi[0,1],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]])
    
    F10 = np.array([
            np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])*(1-u[2])*beta*SdivN[1][0]*Phi[1,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]])

    # Then compute V
    V00 = np.array([[sig,0,0,0,0],
                [0,rhoA,0,0,0],
                [0,0,rhoY,0,0],
                [0,0,0,gamA,0],
                [0,0,0,0,(1-Pi[0])*gamY+Pi[0]*eta]])
    
    V11 = np.array([[sig,0,0,0,0],
                [0,rhoA,0,0,0],
                [0,0,rhoY,0,0],
                [0,0,0,gamA,0],
                [0,0,0,0,(1-Pi[1])*gamY+Pi[1]*eta]])   

    V10 = np.zeros((5,5))
    V01 = np.copy(V10)
    
    F = np.bmat([[F00,F01],[F10,F11]])
    V = np.bmat([[V00,V01],[V10,V11]])
       
    V_inv = np.linalg.inv(V)
    Prod = F*V_inv

    # Reproduction number is the largest eigenvalue of the next generation matrix
    rho = np.amax(abs(np.linalg.eigvals(Prod)))

    return rho

################
### Used to find the optimal control measures for a specific R0 / Cost
################

def optimalControls_R(dictVar,uVec=[0.0,0.0,0.0,0.0],rho = 1.0,\
                      immunity = 0.0):
    '''
    Return: 
        opt_uVec: optimal control values
        minCost: Minimum cost based on 'opt_uVec'

    Parameters:
        dictVar: Dictionary with variables from initialization() 
        uVec: Array of controls; initial guess
           [0]: Low risk testing rate ( 0-1 )
           [1]: High risk resting rate ( 0-1 )
           [2]: Low risk distance rate ( 0-1 )
           [3]: High risk distance rate ( 0-1 )
        rho: Reproduction Number
        immunity: (0-1) Immunity Leve
    '''

    # THINGS I NEED
    uMax = dictVar['uMax']
    hiRiskCtrl = dictVar['hiRiskCtrl']
    
    try:

        # rho - computedRho = 0
        consR = {'type':'eq',
                'fun':lambda x,var,imm: rho - compReproductionNumber(x,var,imm),
                'args':[dictVar,immunity]}

        boundary = [ [0,uMax[0]], 
                    [0,uMax[1]],
                    [0,uMax[2]], 
                    [0,uMax[3]] ]
       
        if len(hiRiskCtrl)==2 :
                        
            
            # Try first to solve with no low-risk

           # Constrain low-risk control to 0
            consU0 = {'type':'eq',
                'fun':lambda x: x[0]}
            consU2 = {'type':'eq',
                'fun':lambda x: x[2]}

            soln = opt.minimize(compCost,[0,uMax[1],0,uMax[3]],
            args = dictVar,
            method = 'SLSQP',
            bounds = boundary,
            constraints=[consR,consU0,consU2],
            options = {'disp':False,
                        'eps':0.001})
            
            # If solution has the max hi-risk control, then let the low risk be nonzero
            if (soln.x[1] >= uMax[1]-1.E-8) and (soln.x[3] >= uMax[3] - 1.E-8):

                # Constrain hi-risk controls
                consU1 = {'type':'eq',
                    'fun':lambda x: x[1]-uMax[1]}
                consU3 = {'type':'eq',
                    'fun':lambda x: x[3]-uMax[3]}

                # Add margin in limits in case there is problems with opt.
                soln = opt.minimize(compCost,[uMax[0],uMax[1]+.001,uMax[2],uMax[3]+.001],
                args = dictVar,
                method = 'SLSQP',
                bounds = boundary,
                constraints=[consR,consU1,consU3],
                options = {'disp':False,
                            'eps':0.001})
                

        else:
            soln = opt.minimize(compCost,uVec,
            args = dictVar,
            method = 'SLSQP',
            bounds = boundary,
            constraints=[consR],
            options = {'disp':False,
                        'eps':0.001})

        # soln.x : OPTIMAL uVec SOLUTION
        # soln.fun: MINIMUM COST BASED ON THE OPTIMAL CONTROLS
        return soln.x , soln.fun

    except ValueError:

        # `x0` violates bound constraints"
        uCtrl = uMax

        return uCtrl, compCost(uCtrl,dictVar)


    except Exception as e:

        # STATE THE PARAMETERS THAT CAUSED THE ERROR
        print('\n--------- ERROR ---------')
        print('Function: optimalControls_R')
        print('uVec:',uVec)
        print('rho:',rho)
        print('immunity:',immunity)

        # PRINT ERROR MESSAGE
        print('\n--------- MESSAGE ---------')
        print(e)
        print('-------------------------\n')


def optimalControls_Budget(dictVar,uVec=[0.0,0.0,0.0,0.0], cost = 1.0E10,immunity = 0.0):
    '''
    Return: 
        opt_uVec: optimal control values
        minCost: Minimum cost based on 'opt_uVec'

    Parameters:
        dictVar: Dictionary with variables from initialization() 
        uVec: Array of controls; initial guess
           [0]: Low risk testing rate ( 0-1 )
           [1]: High risk resting rate ( 0-1 )
           [2]: Low risk distance rate ( 0-1 )
           [3]: High risk distance rate ( 0-1 )
        cost: Constant Cost for operations
        immunity: (0-1) Immunity Level
    '''

    # THINGS I NEED
    uMax = dictVar['uMax']
    hiRiskCtrl = dictVar['hiRiskCtrl']

    try:

        # Ensure maxCost >= compCost
        consCost = {'type':'ineq',
                'fun':lambda x,var: cost - compCost(x,var),
                'args':[dictVar]}

        boundary = np.array([ [0,uMax[0]], 
                     [0,uMax[1]],
                     [0,uMax[2]], 
                     [0,uMax[3]] ])
        

        if len(hiRiskCtrl)==2 :

            # Try first to solve with no lo-risk
           # Constrain low-risk control to 0
            consU0 = {'type':'eq',
                'fun':lambda x: x[0]}
            consU2 = {'type':'eq',
                'fun':lambda x: x[2]}

            soln = opt.minimize(compReproductionNumber,[0,uMax[1],0,uMax[3]],
                        args = (dictVar,immunity),
                        method = 'SLSQP',
                        bounds = boundary,
                        constraints=[consCost,consU0,consU2],
                        options = {'disp':False,
                                    'eps':0.001})

            # If solution has the max hi-risk control, then let the low risk be nonzero
            # and ensure that the hi control is maxed out
            if (soln.x[1] >= uMax[1]-1.E-8) and (soln.x[3] >= uMax[3]-1.E-8):
                consU1 = {'type':'eq',
                    'fun':lambda x: x[1]-uMax[1]}
                consU3 = {'type':'eq',
                    'fun':lambda x: x[3]-uMax[3]}


                soln = opt.minimize(compReproductionNumber,uMax,
                        args = (dictVar,immunity),
                        method = 'SLSQP',
                        bounds = boundary,
                        constraints=[consCost,consU1,consU3],
                        options = {'disp':False,
                                    'eps':0.001})

        else:
            soln = opt.minimize(compReproductionNumber,uVec,
                        args = (dictVar,immunity),
                        method = 'SLSQP',
                        bounds = boundary,
                        constraints=[consCost],
                        options = {'disp':False,
                                    'eps':0.001})

        # soln.x : OPTIMAL uVec SOLUTION
        # soln.fun: MINIMUM R0 BASED ON THE OPTIMAL CONTROLS
        return soln.x , compCost(soln.x,dictVar)

    except ValueError:

        # `x0` violates bound constraints"
        return uMax, compCost(uMax,dictVar)

    except Exception as e:

        # STATE THE PARAMETERS THAT CAUSED THE ERROR
        print('\n--------- ERROR ---------')
        print('Function: optimalControls_Cost')
        print('uVec:',uVec)
        print('cost:',cost)
        print('immunity:',immunity)

        # PRINT ERROR MESSAGE
        print('\n--------- MESSAGE ---------')
        print(e)
        print('-------------------------\n')



########
## Used For Simulations
##########

def optimalMinCost_R(dictVar, ti, frac, at = 10 , rt = 0.001):
    '''
    Parameters:
        dictVar: Dictionary with variables from initialization()
        ti: Day in which controls start ( must be > 0 )
        tf: Final Day ( must be > ti)
        frac: fraction of the R0 that is being targetted
        at: ? Absolute total
        rt: ? Relative total

    Purpose:
        Calculate the minimal cost for targeting R0 from t0 to tf days,
        given that no controls are used before ti. 
    '''

    ##################
    # GRAB THE VARIABLES NEEDED FROM THE DICTIONARY 
    ##################
    N = dictVar['N']
    xI = dictVar['xI']
    a = dictVar['a']
    b = dictVar['b']
    c = dictVar['c']
    d = dictVar['d']
    e = dictVar['e']
    f = dictVar['f']
    params = dictVar['params']
    rFrac = dictVar['rFrac']
    uMax = dictVar['uMax']
    tf = dictVar['tf']
    nStep = dictVar['nStep']
    minInfected = dictVar['minInfected']


    # DERIVATIVE STEPS
    dt = tf / nStep

    # THIS SHOULD BE AN INT
    nt = int(ti / dt)


    # CONTROL VALUES ON INTEGRATION STEP
    #uCtrl = np.copy(uZero)
    uCtrlInt = np.zeros( (nt,4) )

    try:
        ## USE FORWARDSOLVE() TO FIND THE SOLUTION 
        ## WITH ONLY HI CONTROLS UP TO ti

        xTmp, _ , _ , allCostSummedTot = forwardSolve(xI, 0, nt, uCtrlInt, dt, a, b, c, d, e, f, params, at, rt  )
                                                    #(xI, jt,nt, uTmp,  dt, a, b, c, d, e, f, params, at, rt
        
        # CONTROLS USED FOR EVERY DAY
        uCtrlDays = np.zeros( (4,tf) )

        # CONTROL VALUES 
        uCtrl = np.zeros(4)

        # Solution (for return)
        xTot = np.zeros((tf,xTmp.shape[1]))
        xTot[:nt+1,:] = xTmp
        
        # Generate the rest of the solution
        for jt in range(nt,nStep-1,1):

            # STATE OF THE SYSTEM AT TIME 'jt'
            xNow = xTot[jt,:]

            # NUMBER OF INDIVIDUALS INFECTED
            threshold = np.sum(xNow[1:5]) + np.sum(xNow[10:15]) 

            if threshold > minInfected:
                
                # COMPUTE THE IMMUNITY LEVEL ( for low and high)
                imm = 1 - xNow[ [0,9] ] / N
                # CURRENT R0 with no control
                curR = compReproductionNumber(uCtrl,dictVar,imm)
                
                # Target R, 2 options
                if rFrac: # Make target R as fraction    
                    targetR = curR * frac
                else:     # Make target R as given value
                    targetR  = frac
                
                # OPTIMAL CONTROLS FOR RESPECTIVE TARGET R0 
                uCtrl, _ = optimalControls_R(dictVar, uVec = uMax,rho=targetR, immunity=imm)
            else:
                # SET CONTROLS TO ZERO
                uCtrl = np.zeros(4)

            # FORWARD SOLVE REQUIRES A LIST OF LIST OF CONTROLS 
            uCtrlInt = np.array([uCtrl])  #np.tile(uCtrl.reshape((4,1)), nt).T
        
            # FIND THE SOLUTION WITH OPTIMAL CONTROLS IN SMALL INTERVALS
            xNow, _ , _ , allCostSumTotTmp = forwardSolve(xNow, jt, jt +1, uCtrlInt, dt, a, b, c, d, e, f, params, at, rt  )

            # Add latest solution to total solution
            xTot[jt+1,:] = xNow[-1,:]
            # Update costs, controls
            allCostSummedTot += np.array(allCostSumTotTmp)
            uCtrlDays[:,jt] = uCtrl

        return xTot, allCostSummedTot, uCtrlDays

    except Exception as e:


        # STATE THE PARAMETERS THAT CAUSED THE ERROR
        print('\n--------- ERROR ---------')
        print('Function: optimalMinCost_R')
        print('ti:',ti)
        print('tf:',tf)
        print('frac:',frac)

        # PRINT ERROR MESSAGE
        print('\n--------- MESSAGE ---------')
        print(e)
        print('-------------------------\n')


def optimalMinCost_Budget(dictVar, ti, maxCost, at = 10 , rt = 0.001):
    '''
    Parameters:
        dictVar: Dictionary with variables from initialization() 
        ti: Day in which controls start ( must be > 0 )
        tf: Final Day ( must be > ti)
        maxCost: Maximum amount of spending
        at: ? Absolute total
        rt: ? Relative total
        nStep: Number of steps for integration

    Purpose:
        Calculate the controls that minimal R0 for max Cost from t0 to tf days,
        given that no controls are used before ti. 
    '''

    ##################
    # GRAB THE VARIABLES NEEDED FROM THE DICTIONARY 
    ##################
    N = dictVar['N']
    xI = dictVar['xI']
    a = dictVar['a']
    b = dictVar['b']
    c = dictVar['c']
    d = dictVar['d']
    e = dictVar['e']
    f = dictVar['f']
    uMax = dictVar['uMax']
    params = dictVar['params']
    tf = dictVar['tf']
    nStep = dictVar['nStep']
    minInfected = dictVar['minInfected']

    # DERIVATIVE STEPS
    dt = tf / nStep

    # THIS SHOULD BE AN INT
    nt = int(ti / dt)

    # CONTROL VALUES ON INTEGRATION STEP
    #uCtrl = np.copy(uZero)
    uCtrlInt = np.zeros( (nt,4) )

    try:
        ## USE FORWARDSOLVE() TO FIND THE SOLUTION 
        ## WITH NO CONTROLS UP TO ti

        xTmp, _ , _ , allCostSummedTot = forwardSolve(xI, 0, nt, uCtrlInt, dt, a, b, c, d, e, f, params, at, rt  )
                                                    #(xI, jt,nt, uTmp,  dt, a, b, c, d, e, f, params, at, rt

        # CONTROLS USED FOR EVERY DAY
        uCtrlDays = np.zeros( (4,tf) )
        # Array of populations
        xTot = np.zeros((tf,xTmp.shape[1]))
        xTot[:nt+1,:] = xTmp
        
        # CONTROL VALUES 
        uCtrl = np.zeros(4)

        for jt in range(nt,nStep-1,1):

            # STATE OF THE SYSTEM AT TIME 'jt'
            xNow = xTot[jt,:]

            # NUMBER OF INDIVIDUALS INFECTED
            threshold = np.sum(xNow[1:5]) + np.sum(xNow[10:15]) 

            if threshold > minInfected:
                
                # COMPUTE THE IMMUNITY LEVEL ( for low and high)
                imm = 1 - xNow[ [0,9] ] / N

                # OPTIMAL CONTROLS THAT MINIMIZE R0 USING ONLY 0 - maxCost DOLLARS
                uCtrl, _ = optimalControls_Budget(dictVar, uVec=uMax,cost = maxCost, immunity=imm)

            else:
                # SET CONTROLS TO ZERO
                uCtrl = np.zeros(4)

            # FORWARD SOLVE REQUIRES A LIST OF LIST OF CONTROLS 
            uCtrlInt = np.array([uCtrl]) 
        
            # FIND THE SOLUTION WITH OPTIMAL CONTROLS IN SMALL INTERVALS
            xNow, _ , _ , allCostSumTotTmp = forwardSolve(xNow, jt, jt +1, uCtrlInt, dt, a, b, c, d, e, f, params, at, rt  )
            xTot[jt+1,:]=xNow[-1,:]
            
            allCostSummedTot += np.array(allCostSumTotTmp)

            uCtrlDays[:,jt] = uCtrl

        return xTot, allCostSummedTot, uCtrlDays

    except Exception as e:


        # STATE THE PARAMETERS THAT CAUSED THE ERROR
        print('\n--------- ERROR ---------')
        print('Function: optimalMinCost_Cost')
        print('ti:',ti)
        print('tf:',tf)
        print('maxCost:',maxCost)

        # PRINT ERROR MESSAGE
        print('\n--------- MESSAGE ---------')
        print(e)
        print('-------------------------\n')

       

#################
### PLOTS 
#################

def optimalMinCost_R_Simulation(dictVar,t1,tstep,f0,ff,fstep,**kwargs):
    '''
    Parameters:
        dictVar: Dictionary with variables from initialization()
        t1: Last Day in which no controls are used ( must be > 0)
        tstep: interval steps from 0 to t1 
        f0: initial fraction of R0
        ff: final fraction of R0
        fstep: interval steps from f0 to ff
        **kwargs: Arguments needed for solution function
            at:
            rt:
    '''

    tf = dictVar['tf'] # Final time
    
    tArray = np.arange(0, t1 , tstep)    
    fracArray = np.arange(f0,ff ,fstep)

    # ALL COST SUMMED ARRAY
    ACSArr = np.zeros( ( len(tArray), len(fracArray), 8 ) )

    nRuns = len(tArray)*len(fracArray)
    curRun = 1

    for ii, ti in enumerate(tArray):

        for kk, frac in enumerate(fracArray):
            print("R RUN: {} out of {}".format(curRun,nRuns))
            curRun+=1

            _,ACSArr[ii,kk,:],_ = optimalMinCost_R(dictVar, ti, frac, *kwargs)

    ###############
    # SAVE DATA 
    ###############

    data = {'tArray':tArray.tolist(),
            'fracArray':fracArray.tolist(),
            'data': ACSArr.tolist() }

    #filename = 'optimalMinCost_R_Simulation.json'
    filename = dictVar['filename'][0]


    file_path = os.path.join('./', filename)

    with open(file_path, 'w') as f:
        json.dump(data, f,indent = 4)


def optimalMinCost_R_plots(dictVar):
    
    
    filename = dictVar['filename'][0]
    totCostLevels = dictVar['totCostLevels']
    deathLevels = dictVar['deathLevels']
    nLevel = dictVar['nLevel']
    rFrac = dictVar['rFrac']
    showAllPlots = dictVar['showAllPlots']
    
    ############################################

    #filename = 'optimalMinCost_R_Simulation.json'
    filePath = os.path.join('./', filename)

    with open(filePath,'r') as f:
        data = json.load(f)

    tArray = data['tArray']
    fracArray = data['fracArray']
    ACSArr = np.array(data['data'])

    ##########################################



    ########################################
    ### PLOTS
    ########################################
    # plots of all costs (optional)
    if showAllPlots:
        plt.figure(1) # PLOT FOR MARIGINAL COST
    
        implementCost  = ACSArr[:,:,0] + ACSArr[:,:,1]  + ACSArr[:,:,2] + ACSArr[:,:,3] 
        pos1 = plt.contourf(tArray,
                            fracArray,
                            implementCost.T,
                            levels = nLevel) # Note: Transpose because X needs to be in the column
    
        cbar1 = plt.colorbar(pos1) 
        cbar1.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Fraction $R_e$')
        plt.title('Implementation marginal cost')
    
        ########################################
        plt.figure(2) # PLOT FOR SICKNESS COST
        pos2 = plt.contourf(tArray,
                            fracArray,
                            ACSArr[:,:,4].T,
                            levels = nLevel) # Note: Transpose because X needs to be in the column
    
        cbar2 = plt.colorbar(pos2) 
        cbar2.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Fraction $R_e$')
        plt.title('Sickness Cost')   
    
        ########################################
        plt.figure(3) # PLOT FOR HOSPITAL COST
    
        pos3 = plt.contourf(tArray,
                            fracArray,
                            ACSArr[:,:,5].T,
                            levels = nLevel) # Note: Transpose because X needs to be in the column
    
        cbar3 = plt.colorbar(pos3) 
        cbar3.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Fraction $R_e$')
        plt.title('Hospital Cost')
    
        ########################################
        plt.figure(4) # PLOT FOR DEATH COST
        pos4 = plt.contourf(tArray,
                            fracArray,
                            ACSArr[:,:,6].T,
                            levels = nLevel) # Note: Transpose because X needs to be in the column
    
        cbar4 = plt.colorbar(pos4) 
        cbar4.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Fraction $R_e$')
        plt.title('Number of Deaths')
    
        ########################################
        plt.figure(5) # PLOT FOR REMAIN INFECTED COST
        pos5 = plt.contourf(tArray,
                            fracArray,
                            ACSArr[:,:,7].T,
                            levels = nLevel) # Note: Transpose because X needs to be in the column
    
        cbar5 = plt.colorbar(pos5) 
        cbar5.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Fraction $R_e$')
        plt.title('Number of remaining infections')

    ########################################
    plt.figure(6) # PLOT FOR MARIGINAL COST W/ DEATH CONTOURS

    ## Smoothing    
    tmp  = ACSArr[:,:,0] + ACSArr[:,:,1]  + ACSArr[:,:,2] + ACSArr[:,:,3] 
    implementCost = 1.*tmp
    implementCost[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
                                                    
    pos6 = plt.contourf(tArray,
                        fracArray,
                        np.minimum(implementCost,totCostLevels[-1]).T,
                        levels = totCostLevels) # Note: Transpose because X needs to be in the column


    tmp = 1.*ACSArr[:,:,6]
    nDeaths = 1.*tmp
    nDeaths[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                           tmp[:-2,2:] + tmp[1:-1,:-2] +
                           tmp[1:-1,2:] + tmp[2:,:-2] +
                           tmp[2:,1:-1] + tmp[2:,2:] +
                           4*tmp[1:-1,1:-1])/12                                
    
    pos7 = plt.contour(tArray,
                    fracArray,
                    nDeaths.T,
                    colors = 'white',
                    levels = deathLevels) # Note: Transpose because X needs to be in the column


    plt.clabel(pos7,fmt=lambda x: "{:.1E}".format(x))


    cbar1 = plt.colorbar(pos6)
    cbar1.set_label('Policy cost in billions of USD')

    plt.xlabel('Control policy start day')
    if rFrac:
        plt.ylabel('Target  $R_e$ fraction')
        plt.title("Constant $R_e$ reduction strategies")
    else:
       plt.ylabel('Target  $R_e$ value')
       plt.title("$R_e$ target value strategies")
    

    ###############################
    # FIND THE MINIMAL COST FOR A GIVEN DEATH COST

    blackDots = [ [] , [] ]

    for j,conLvl in enumerate(pos7.levels):

        # ENSURES THAT THE CONTOUR IS NOT EMPTY
        if len(pos7.allsegs[j]) > 0:

            # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
            contoursX =  np.array(measure.find_contours(nDeaths, conLvl,'low')) 
            contours = np.concatenate( (contoursX) )
            
            # Interpolation routine
            tmp = np.zeros(len(contours))            
            # Loop through the arrays 
            for k, xy in enumerate(contours):
                [xx,yy] = np.floor(xy).astype(int)
                xx = min([xx,len(tArray)-2])
                yy = min([yy,len(fracArray)-2])
                
                tmpMx = np.array([[1,xx,yy,xx*yy],
                                    [1,xx+1,yy,(xx+1)*yy],
                                    [1,xx,yy+1,xx*(yy+1)],
                                    [1,xx+1,yy+1,(xx+1)*(yy+1)]])
                tmpVec = np.array([implementCost[xx,yy],
                                implementCost[xx+1,yy],
                                implementCost[xx,yy+1],
                                implementCost[xx+1,yy+1]])
                coeffs = np.linalg.solve(tmpMx,tmpVec)
                tmp[k] = coeffs[0] + coeffs[1]*xy[0] + coeffs[2]*xy[1] \
                            + coeffs[3]*xy[0]*xy[1]
            
            # FIND THE SMALLEST COST
            minIdx = np.argmin(tmp)

            # FIND THE INCEDES FROM THE CONTOUR POINTS THAT HAD THE SMALLEST COST
            # THIS IS MORE OF AN ESTIMATE THAN AN EXACT FACT
            curIdx = contours[minIdx]
            # MAP THE INDEX TO THE X,Y AXIS VALUE
            tPoint = curIdx[0] * (tArray[1] - tArray[0]) 
            fPoint = curIdx[1] * (fracArray[1] - fracArray[0]) + fracArray[0]

            # STORE THE POINTS 
            blackDots[0].append( tPoint )
            blackDots[1].append( fPoint )


    plt.scatter( blackDots[0], blackDots[1] , s = 100,marker = '*', color = 'k', zorder = 2)


    ###############################



    plt.show()


########

def optimalMinCost_Budget_Simulation(dictVar,t1,tstep,minCost,maxCost,cStep,**kwargs):
    '''
    Parameters:
        dictVar: Dictionary with variables from initialization() 
        t1: Last Day in which no controls are used ( must be > 0)
        tf: Final Day ( must be > t1)
        tstep: interval steps from 0 to t1 
        minCost: Minimum amount to spend
        maxCost: Maximum amount to spend
        cStep: interval steps from minCost to maxCost
        **kwargs:
            at:
            rt:
            nStep: Number of steps for integration
    '''
    tArray = np.arange(0, t1 , tstep)    
    costArray = np.arange(minCost,maxCost,cStep)

    # ALL COST SUMMED ARRAY
    ACSArr = np.zeros( ( len(tArray), len(costArray), 8 ) )

    nRuns = len(tArray)*len(costArray)
    curR = 1


    for ii, ti in enumerate(tArray):

        for kk, curCost in enumerate(costArray):
            print("C RUN: {} out of {}".format(curR,nRuns))
            curR+=1

            _,ACSArr[ii,kk,:],_ = optimalMinCost_Budget(dictVar, ti, curCost, *kwargs)


    ###############
    # SAVE DATA 
    ###############

    data = {'tArray':tArray.tolist(),
            'costArray':costArray.tolist(),
            'data': ACSArr.tolist() }
    
    
    filename = dictVar['filename'][0]
    #filename = 'optimalMinCost_Budget_Simulation.json'

    file_path = os.path.join('./', filename)

    with open(file_path, 'w') as f:
        json.dump(data, f,indent = 4)


def optimalMinCost_Budget_plots(dictVar):


    ############################################
    filename = dictVar['filename'][0]
    #filename = 'optimalMinCost_Budget_Simulation.json'
    totCostLevels = dictVar['totCostLevels']
    deathLevels = dictVar['deathLevels']
    showAllPlots = dictVar['showAllPlots']
    
    filePath = os.path.join('./', filename)

    with open(filePath,'r') as f:
        data = json.load(f)

    tArray = data['tArray']
    costArray = data['costArray']
    ACSArr = np.array(data['data'])

    ##########################################


    ########################################
    ### PLOTS
    ########################################
    if showAllPlots:
        plt.figure(1) # PLOT FOR MARIGINAL COST
    
        implementCost  = ACSArr[:,:,0] + ACSArr[:,:,1]  + ACSArr[:,:,2] + ACSArr[:,:,3] 
        pos1 = plt.contourf(tArray,
                            costArray,
                            implementCost.T,
                            levels = 15) # Note: Transpose because X needs to be in the column
    
        cbar1 = plt.colorbar(pos1) 
        cbar1.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Daily Budget  ($)')
        plt.title('Implementation Marginal Cost')
    
    
        ########################################
        ### PLOTS
        ########################################
        plt.figure(2) # PLOT FOR SICKNESS COST
        pos2 = plt.contourf(tArray,
                            costArray,
                            ACSArr[:,:,4].T,
                            levels = 15) # Note: Transpose because X needs to be in the column
    
        cbar2 = plt.colorbar(pos2) 
        cbar2.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Daily Budget  ($)')
        plt.title('Sickness Cost')   
    
        ########################################
        plt.figure(3) # PLOT FOR HOSPITAL COST
    
        pos3 = plt.contourf(tArray,
                            costArray,
                            ACSArr[:,:,5].T,
                            levels = 15) # Note: Transpose because X needs to be in the column
    
        cbar3 = plt.colorbar(pos3) 
        cbar3.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Daily Budget  ($)')
        plt.title('Hospital Cost')
    
        ########################################
        plt.figure(4) # PLOT FOR DEATH COST
        pos4 = plt.contourf(tArray,
                            costArray,
                            ACSArr[:,:,6].T,
                            levels = 15) # Note: Transpose because X needs to be in the column
    
        cbar4 = plt.colorbar(pos4) 
        cbar4.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Daily Budget  ($)')
        plt.title('Death Cost')
    
        ########################################
        plt.figure(5) # PLOT FOR REMAIN INFECTED COST
        pos5 = plt.contourf(tArray,
                            costArray,
                            ACSArr[:,:,7].T,
                            levels = 15) # Note: Transpose because X needs to be in the column
    
        cbar5 = plt.colorbar(pos5) 
        cbar5.set_label('Cost (US Dollars)')
    
        plt.xlabel('Policy Start Day of Control Measures')
        plt.ylabel('Daily Budget  ($)')
        plt.title('Remain Infected Cost')
    

    ########################################
    plt.figure(6) # PLOT FOR TOTAL COST W/ DEATH CONTOURS

    implementCost  = ACSArr[:,:,0] + ACSArr[:,:,1]  + ACSArr[:,:,2] + ACSArr[:,:,3] 
    pos6 = plt.contourf(tArray,
                        costArray,
                        np.minimum(implementCost,totCostLevels[-1]).T,
                        levels = totCostLevels) # Note: Transpose because X needs to be in the column

    pos7 = plt.contour(tArray,
                    costArray,
                    ACSArr[:,:,6].T,
                    colors = 'white',
                    levels = deathLevels) # Note: Transpose because X needs to be in the column


    plt.clabel(pos7,fmt=lambda x: "{:.1E}".format(x))


    cbar1 = plt.colorbar(pos6) 
    cbar1.set_label('Control cost in billions of USD')

    plt.xlabel('Control policy start day')
    plt.ylabel('Daily budget (tens of millions of USD) ')
    plt.title("Fixed Daily Cost Ceiling Policies")
    plt.ylim([0,5.E7])

    ###############################
    # FIND THE MINIMAL COST FOR A GIVEN DEATH COST

    nDeaths = ACSArr[:,:,6]

    blackDots = [ [] , [] ]

    for j,conLvl in enumerate(pos7.levels):

        # ENSURES THAT THE CONTOUR IS NOT EMPTY
        if len(pos7.allsegs[j]) > 0:

            # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
            # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
            contoursX =  np.array(measure.find_contours(nDeaths, conLvl,'low')) 
            contours = np.concatenate( (contoursX) )

            # Interpolation routine
            tmp = np.zeros(len(contours))            
            # Loop through the arrays 
            for k, xy in enumerate(contours):
                [xx,yy] = np.floor(xy).astype(int)
                xx = min([xx,len(tArray)-2])
                yy = min([yy,len(costArray)-2])
                
                tmpMx = np.array([[1,xx,yy,xx*yy],
                                    [1,xx+1,yy,(xx+1)*yy],
                                    [1,xx,yy+1,xx*(yy+1)],
                                    [1,xx+1,yy+1,(xx+1)*(yy+1)]])
                tmpVec = np.array([implementCost[xx,yy],
                                implementCost[xx+1,yy],
                                implementCost[xx,yy+1],
                                implementCost[xx+1,yy+1]])
                coeffs = np.linalg.solve(tmpMx,tmpVec)
                tmp[k] = coeffs[0] + coeffs[1]*xy[0] + coeffs[2]*xy[1] \
                            + coeffs[3]*xy[0]*xy[1]
                
            # FIND THE SMALLEST COST
            minIdx = np.argmin(tmp)

            # FIND THE INCEDES FROM THE CONTOUR POINTS THAT HAD THE SMALLEST COST
            # THIS IS MORE OF AN ESTIMATE THAN AN EXACT FACT
            curIdx = contours[minIdx]
            # MAP THE INDEX TO THE X,Y AXIS VALUE
            tPoint = curIdx[0] * (tArray[1] - tArray[0]) 
            cPoint = curIdx[1] * (costArray[1] - costArray[0]) + costArray[0]

            # STORE TEH POINTS 
            blackDots[0].append( tPoint )
            blackDots[1].append( cPoint )


    plt.scatter( blackDots[0], blackDots[1] , s = 100,marker = '*', color = 'k', zorder = 2)



    plt.show()


########
def findCostForGivenDeath(implementCostArr,nDeathsArr,xArray,yArray,deathValue):
    '''
    Returns: A 2-element list of values where the minimum cost was found for a given death value
    [0]: The Row Value (list)
    [1]: implementation cost (for each row value)

    Parameter:
        implementCostArr: Expecting a 2D array
        nDeathArr: expecting a 2D array
        xArray: row array for the 2D arrays
        yArray: col array for the 2D arrays
        deathValues: a list of values in which a contour line is to be found (one value)
    '''

    implementCostForGivenDeath = [  [] , [] ]


    # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
    contoursX = np.array(measure.find_contours(nDeathsArr, deathValue,'low'))
    if len(contoursX) > 0:

        # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
        contours = np.concatenate( (contoursX) ) 
        # Interpolation routine
        tmp = np.zeros(len(contours))            
        # Loop through the arrays 
        for k, xy in enumerate(contours):
            [xx,yy] = np.floor(xy).astype(int)
            xx = min([xx,len(xArray)-2])
            yy = min([yy,len(yArray)-2])
            
            tmpMx = np.array([[1,xx,yy,xx*yy],
                            [1,xx+1,yy,(xx+1)*yy],
                            [1,xx,yy+1,xx*(yy+1)],
                            [1,xx+1,yy+1,(xx+1)*(yy+1)]])
            tmpVec = np.array([implementCostArr[xx,yy],
                            implementCostArr[xx+1,yy],
                            implementCostArr[xx,yy+1],
                            implementCostArr[xx+1,yy+1]])
            coeffs = np.linalg.solve(tmpMx,tmpVec)
            tmp[k] = coeffs[0] + coeffs[1]*xy[0] + coeffs[2]*xy[1] \
                    + coeffs[3]*xy[0]*xy[1]
        
            implementCostForGivenDeath[0].append(xx)
            implementCostForGivenDeath[1].append(tmp[k])


    return implementCostForGivenDeath
    

def findDeathForGivenCost(implementCostArr,nDeathsArr,xArray,yArray,costValue):
    '''
    Returns: A 2-element list of values where the minimum death was found for a given cost value
    [0]: The Row Value (list)
    [1]:Number of death (for each row value)

    Parameter:
        implementCostArr: Expecting a 2D array
        nDeathArr: expecting a 2D array
        xArray: row array for the 2D arrays
        yArray: col array for the 2D arrays
        deathValues: value in which a contour line is to be found ( one value)
    '''

    deathForGivenCost = [  [] , []]


    # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
    contoursX = np.array(measure.find_contours(implementCostArr, costValue,'low'))
    if len(contoursX) > 0:

        # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
        contours = np.concatenate( (contoursX) )
        # Interpolation routine
        tmp = np.zeros(len(contours))            
        # Loop through the arrays 
        for k, xy in enumerate(contours):
            [xx,yy] = np.floor(xy).astype(int)
            xx = min([xx,len(xArray)-2])
            yy = min([yy,len(yArray)-2])
            
            tmpMx = np.array([[1,xx,yy,xx*yy],
                            [1,xx+1,yy,(xx+1)*yy],
                            [1,xx,yy+1,xx*(yy+1)],
                            [1,xx+1,yy+1,(xx+1)*(yy+1)]])
            tmpVec = np.array([nDeathsArr[xx,yy],
                            nDeathsArr[xx+1,yy],
                            nDeathsArr[xx,yy+1],
                            nDeathsArr[xx+1,yy+1]])
            coeffs = np.linalg.solve(tmpMx,tmpVec)
            tmp[k] = coeffs[0] + coeffs[1]*xy[0] + coeffs[2]*xy[1] \
                    + coeffs[3]*xy[0]*xy[1]
        
            deathForGivenCost[0].append(xx)
            deathForGivenCost[1].append(tmp[k])


    return deathForGivenCost
    
###################

def findMinCostForGivenDeath(implementCostArr,nDeathsArr,xArray,yArray,deathValue):
    '''
    Returns: A 4-element list of values where the minimum cost was found for a given death value
    [0]: The Row Value
    [1]: The column value
    [2]: Number of Death
    [3]: Implementation Cost

    Parameter:
        implementCostArr: Expecting a 2D array
        nDeathArr: expecting a 2D array
        xArray: row array for the 2D arrays
        yArray: col array for the 2D arrays
        deathValues: a list of values in which a contour line is to be found
    '''


    optVals = [ [], [], [], [] ]

    for contourLvl in deathValue :

        # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
        contoursX = np.array(measure.find_contours(nDeathsArr, contourLvl,'low'))
        #print('lvl:{} #Contours: {}'.format(contourLvl,len(contoursX)))
        if len(contoursX) > 0:

            # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
            contours = np.concatenate( (contoursX) )
            # Interpolation routine
            tmp = np.zeros(len(contours))            
            # Loop through the arrays 
            for k, xy in enumerate(contours):
                [xx,yy] = np.floor(xy).astype(int)
                xx = min([xx,len(xArray)-2])
                yy = min([yy,len(yArray)-2])
                
                tmpMx = np.array([[1,xx,yy,xx*yy],
                                [1,xx+1,yy,(xx+1)*yy],
                                [1,xx,yy+1,xx*(yy+1)],
                                [1,xx+1,yy+1,(xx+1)*(yy+1)]])
                tmpVec = np.array([implementCostArr[xx,yy],
                                implementCostArr[xx+1,yy],
                                implementCostArr[xx,yy+1],
                                implementCostArr[xx+1,yy+1]])
                coeffs = np.linalg.solve(tmpMx,tmpVec)
                tmp[k] = coeffs[0] + coeffs[1]*xy[0] + coeffs[2]*xy[1] \
                        + coeffs[3]*xy[0]*xy[1]
            
            # FIND THE SMALLEST COST
            minIdx = np.argmin(tmp)

            # FIND THE INDICES FROM THE CONTOUR POINTS THAT HAD THE SMALLEST COST
            # THIS IS MORE OF AN ESTIMATE THAN AN EXACT FACT
            curIdx = contours[minIdx]
            # MAP THE INDEX TO THE X,Y AXIS VALUE
            tPoint = curIdx[0] * (xArray[1] - xArray[0]) + xArray[0]
            cPoint = curIdx[1] * (yArray[1] - yArray[0]) + yArray[0]

            # STORE STUFF
            optVals[0].append( tPoint ) # Row Value
            optVals[1].append( cPoint ) # Col value
            optVals[2].append( contourLvl ) # NUMBER OF DEATHS
            optVals[3].append( tmp[minIdx] ) # IMPLEMENTATION COST

    return np.array(optVals)
    

 
####################

def pareto_plot(dictVar,xDays = 16):
    '''
    Parameter:
        dictVar: Dictionary with variables from initialization()
        xDays: Delay start of control by at least xDays days (>)
    '''


    # dictVars needed
    filenames = dictVar['filename']

    deathLevels = dictVar['deathLevels']

    ### processing for three strategies
    filePath = os.path.join('./', filenames[0])

    with open(filePath,'r') as f:
        data = json.load(f)

    tArrayBudget = np.array( data['tArray'] )
    costArrayBudget = np.array( data['costArray'] )
    ACSArrBudget = np.array(data['data'])   
    implementCostArrBudget = ACSArrBudget[:,:,0] + ACSArrBudget[:,:,1] + ACSArrBudget[:,:,2] + ACSArrBudget[:,:,3]
    nDeathsBudget = ACSArrBudget[:,:,6]



    ####################
    # Smoothing
    tmp = 1.*implementCostArrBudget
    implementCostArrBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsBudget
    nDeathsBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                



    blackDotsBudget = findMinCostForGivenDeath(implementCostArrBudget,
                                               nDeathsBudget,
                                               tArrayBudget,
                                               costArrayBudget,
                                               deathLevels)

    
    # find black stars after X Days
    idx = tArrayBudget > xDays

    blackDotsBudgetAfterX = findMinCostForGivenDeath(implementCostArrBudget[idx],
                                               nDeathsBudget[idx],
                                               tArrayBudget[idx],
                                               costArrayBudget,
                                               deathLevels)



    ####################

    # filename = 'optimalMinCost_rFrac_Basic.json'
    filePath = os.path.join('./', filenames[1])

    with open(filePath,'r') as f:
        data = json.load(f)

    tArrayR0 = np.array( data['tArray'] )
    fracArrayR0 = np.array( data['fracArray'] )
    ACSArrR0 = np.array(data['data'])
    implementCostArrR0 = ACSArrR0 [:,:,0] + ACSArrR0 [:,:,1] + ACSArrR0 [:,:,2] + ACSArrR0 [:,:,3]
    nDeathsR0 = ACSArrR0[:,:,6]

    # Smoothing
    tmp = 1.*implementCostArrR0
    implementCostArrR0[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsR0
    nDeathsR0[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                

    blackDotsR0 = findMinCostForGivenDeath(implementCostArrR0,
                                            nDeathsR0,
                                            tArrayR0,
                                            fracArrayR0,
                                            deathLevels)
    
    # find black stars after X Days
    idx = tArrayR0 > xDays

    blackDotsR0AfterX = findMinCostForGivenDeath(implementCostArrR0[idx],
                                            nDeathsR0[idx],
                                            tArrayR0[idx],
                                            fracArrayR0,
                                            deathLevels)


    ####################

    # filename = 'optimalMinCost_rTgt_Basic.json'
    filePath = os.path.join('./', filenames[2])

    with open(filePath,'r') as f:
        data = json.load(f)

    tArrayRtgt = np.array( data['tArray'] )
    fracArrayRtgt = np.array( data['fracArray'] )
    ACSArrRtgt = np.array(data['data'])
    implementCostArrRtgt = ACSArrRtgt [:,:,0] + ACSArrRtgt [:,:,1] + ACSArrRtgt [:,:,2] + ACSArrRtgt [:,:,3]
    nDeathsRtgt = ACSArrRtgt[:,:,6]

    # Smoothing
    tmp = 1.*implementCostArrRtgt
    implementCostArrRtgt[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsRtgt
    nDeathsRtgt[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                

    blackDotsconstRvars = findMinCostForGivenDeath(implementCostArrRtgt,
                                            nDeathsRtgt,
                                            tArrayRtgt,
                                            fracArrayRtgt,
                                            deathLevels)


    # find black stars after X Days
    idx = tArrayRtgt > xDays

    blackDotsconstRvarsAfterX = findMinCostForGivenDeath(implementCostArrRtgt[idx],
                                            nDeathsRtgt[idx],
                                            tArrayRtgt[idx],
                                            fracArrayRtgt,
                                            deathLevels)

    #####################

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cycle3 = (cycler(color=['royalblue','tomato','lime','blue', 'red', 'green', 'teal','fuchsia','lawngreen']) +
              cycler(linestyle=['-', '-', '-', '--','--','--',':',':',':'])+
              cycler(marker=['o', '+', '*', 'o','+','*','o','+','*']))
    ax.set_prop_cycle(cycle3)

    ax.plot(blackDotsBudget[3],blackDotsBudget[2],label='Daily budget')
    ax.plot(blackDotsR0[3],blackDotsR0[2],label='$R_e$ fraction')
    ax.plot(blackDotsconstRvars[3],blackDotsconstRvars[2],label='Target $R_e$')


 
    # PLOTS FOR > X Days

    ax.plot(blackDotsBudgetAfterX[3],blackDotsBudgetAfterX[2],label='Daily budget >{}'.format(xDays))
    ax.plot(blackDotsR0AfterX[3],blackDotsR0AfterX[2],label='$R_e$ fraction >{}'.format(xDays))
    ax.plot(blackDotsconstRvarsAfterX[3],blackDotsconstRvarsAfterX[2],label='Target $R_e$ >{}'.format(xDays))
    

    #plt.yscale('log')
    ax.set_xlabel(' Control cost in billions of USD ')
    ax.set_ylabel(' Number of Deaths')
    ax.set_title('Pareto optima for 3 strategies')
    ax.set_aspect('auto','box')
    
    xLimits = dictVar['xLimits']
    yLimits = dictVar['yLimits']
    if len(xLimits)==2:
        ax.set_xlim(xLimits[0],xLimits[1])
    if len(yLimits)==2:
        ax.set_ylim(yLimits[0],yLimits[1])
    ax.grid(True)


    plt.legend()
    plt.tight_layout()
    plt.show()


def implementCost_VS_StartDay(dictVar, deathValue = [ 40E3,80E3]):
    ''' 
    Parameter:
        dictVar: Dictionary with variables from initialization()
        deathValue: A list of death values; Used to find the minimal cost for the given death value
    '''

    ########################
    filenames = dictVar['filename']
    # filename = 'optimalMinCost_ConstBudget_Basic.json'
    filePath = os.path.join('./', filenames[0])

    with open(filePath,'r') as f:
        data = json.load(f)

    tArrayBudget = data['tArray']
    costArrayBudget = data['costArray']
    ACSArrBudget = np.array(data['data'])   
    implementCostArrBudget = ACSArrBudget[:,:,0] + ACSArrBudget[:,:,1] + ACSArrBudget[:,:,2] + ACSArrBudget[:,:,3]
    nDeathsBudget = ACSArrBudget[:,:,6]

    ########################
    #filename = 'optimalMinCost_rFrac_Basic.json'
    filePath = os.path.join('./', filenames[1])

    with open(filePath,'r') as f:
        data = json.load(f)

    tArrayR0 = data['tArray']
    fracArrayR0 = data['fracArray']
    ACSArrR0 = np.array(data['data'])
    implementCostArrR0 = ACSArrR0 [:,:,0] + ACSArrR0 [:,:,1] + ACSArrR0 [:,:,2] + ACSArrR0 [:,:,3]
    nDeathsR0 = ACSArrR0[:,:,6]

    ########################
    # filename = 'optimalMinCost_rTgt_Basic.json'
    filePath = os.path.join('./', filenames[2])

    with open(filePath,'rb') as f:
        data = json.load(f)

    tArrayconstRvars = data['tArray']
    fracArrayconstRvars = data['fracArray']
    ACSArrconstRvars = np.array(data['data'])
    implementCostArrconstRvars = ACSArrconstRvars [:,:,0]\
        + ACSArrconstRvars [:,:,1]\
            + ACSArrconstRvars [:,:,2]\
                + ACSArrconstRvars [:,:,3]
    nDeathsconstRvars = ACSArrconstRvars[:,:,6]


    ### Smoothing 
    tmp = 1.*implementCostArrBudget
    implementCostArrBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsBudget
    nDeathsBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*implementCostArrR0
    implementCostArrR0[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsR0
    nDeathsR0[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                

    tmp = 1.*implementCostArrconstRvars
    implementCostArrconstRvars[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsconstRvars
    nDeathsconstRvars[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    
    ########################
    ### PLOTS
    ########################
    fig,ax = plt.subplots()
    cycle3 = (cycler(color=['royalblue','tomato','lime','blue', 'red', 'green', 'teal','fuchsia','lawngreen']) +
              cycler(linestyle=['o:', '+:', '*:', 'o-.','+_.','*-.','o--','+--','*--']))
    ax.set_prop_cycle(cycle3)

    for DOI in deathValue:

        xArrBudget,yArrBudget = findCostForGivenDeath(implementCostArrBudget,
                                        nDeathsBudget,
                                        tArrayBudget,
                                        costArrayBudget,
                                        DOI)
        
        xArrR0,yArrR0 = findCostForGivenDeath(implementCostArrR0,
                                        nDeathsR0,
                                        tArrayR0,
                                        fracArrayR0,
                                        DOI)

        xArrconstRvars,yArrconstRvars = findCostForGivenDeath(implementCostArrconstRvars,
                                        nDeathsconstRvars,
                                        tArrayconstRvars,
                                        fracArrayconstRvars,
                                        DOI)

        deaths = int(DOI//1000)

        # Pick best option for given start date

        budgetIx = np.array(np.lexsort((yArrBudget,xArrBudget))).astype(int)
        xArrBudget = np.array(xArrBudget)
        yArrBudget = np.array(yArrBudget)
        xArrBudget = xArrBudget[budgetIx]
        yArrBudget = yArrBudget[budgetIx]
        [xArrBudget,budgetIx] = np.unique(xArrBudget,return_index = True)
        yArrBudget = yArrBudget[budgetIx.astype(int)]


        R0Ix = np.array(np.lexsort((yArrR0,xArrR0))).astype(int)
        xArrR0 = np.array(xArrR0)
        yArrR0 = np.array(yArrR0)
        xArrR0 = xArrR0[R0Ix]
        yArrR0 = yArrR0[R0Ix]
        [xArrR0,R0Ix] = np.unique(xArrR0,return_index = True)
        yArrR0 = yArrR0[R0Ix.astype(int)]

        constRvarsIx = np.array(np.lexsort((yArrconstRvars,xArrconstRvars))).astype(int)
        xArrconstRvars = np.array(xArrconstRvars)
        yArrconstRvars = np.array(yArrconstRvars)
        xArrconstRvars = xArrconstRvars[constRvarsIx]
        yArrconstRvars = yArrconstRvars[constRvarsIx]
        [xArrconstRvars,constRvarsIx] = np.unique(xArrconstRvars,return_index = True)
        yArrconstRvars = yArrconstRvars[constRvarsIx.astype(int)]


        # Do plots
        # Set color and linestyle cycle
        
        ax.plot(xArrBudget,yArrBudget,'o:',label='Daily budget w/ {}K deaths'.format(deaths))
        ax.plot(xArrR0,yArrR0,'+:',label='$R_e$ fraction w/ {}K deaths'.format(deaths))
        ax.plot(xArrconstRvars,yArrconstRvars,'*:',label='Target $R_e$  w/ {}K deaths'.format(deaths))

    xLimits = dictVar['xLimits']
    yLimits = dictVar['yLimits']
    if len(xLimits)==2:
        ax.set_xlim(xLimits[0],xLimits[1])
    if len(yLimits)==2:
        ax.set_ylim(yLimits[0],yLimits[1])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend()
    ax.grid(True)
    #fig.tight_layout()
    ax.set_title('Effect of control start date on policy costs')
    ax.set_xlabel('Control policy start day')
    ax.set_ylabel('Implementation cost (billions of USD)')

    plt.show()


def Deaths_VS_StartDay(dictVar, costValue = [ 2E9,4E9,6E9]):
    '''
        Parameter:
            dictVar: Dictionary with variables from initialization()
            costValue: Total costs for contours
    '''

    ########################
    filenames = dictVar['filename']

    # filename = 'optimalMinCost_ConstBudget_Basic.json'
    filePath = os.path.join('./', filenames[0])

    with open(filePath,'r') as f:
        data = json.load(f)

    tArrayBudget = data['tArray']
    costArrayBudget = data['costArray']
    ACSArrBudget = np.array(data['data'])   
    implementCostArrBudget = ACSArrBudget[:,:,0] + ACSArrBudget[:,:,1] + ACSArrBudget[:,:,2] + ACSArrBudget[:,:,3]
    nDeathsBudget = ACSArrBudget[:,:,6]

    ########################
    # filename = 'optimalMinCost_rFrac_Basic.json'
    filePath = os.path.join('./', filenames[1])

    with open(filePath,'r') as f:
        data = json.load(f)

    tArrayR0 = data['tArray']
    fracArrayR0 = data['fracArray']
    ACSArrR0 = np.array(data['data'])
    implementCostArrR0 = ACSArrR0 [:,:,0] + ACSArrR0 [:,:,1] + ACSArrR0 [:,:,2] + ACSArrR0 [:,:,3]
    nDeathsR0 = ACSArrR0[:,:,6]

    ########################
    # filename = 'optimalMinCost_rTgt_Basic.json'
    filePath = os.path.join('./', filenames[2])

    with open(filePath,'rb') as f:
        data = json.load(f)

    ACSArrconstRvars = np.array(data['data'])
    tArrayconstRvars = data['tArray']
    fracArrayconstRvars = data['fracArray']
    implementCostArrconstRvars = ACSArrconstRvars [:,:,0]\
        + ACSArrconstRvars [:,:,1]\
            + ACSArrconstRvars [:,:,2]\
                + ACSArrconstRvars [:,:,3]
    nDeathsconstRvars = ACSArrconstRvars[:,:,6]


    ### Smoothing 
    tmp = 1.*implementCostArrBudget
    implementCostArrBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsBudget
    nDeathsBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*implementCostArrR0
    implementCostArrR0[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsR0
    nDeathsR0[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                

    tmp = 1.*implementCostArrconstRvars
    implementCostArrconstRvars[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    tmp = 1.*nDeathsconstRvars
    nDeathsconstRvars[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                tmp[:-2,2:] + tmp[1:-1,:-2] +
                                tmp[1:-1,2:] + tmp[2:,:-2] +
                                tmp[2:,1:-1] + tmp[2:,2:] +
                                4*tmp[1:-1,1:-1])/12                                
    

    ########################
    ### PLOTS
    ########################
    fig,ax = plt.subplots()
    cycle3 = (cycler(color=['royalblue','tomato','lime','blue', 'red', 'green', 'teal','fuchsia','lawngreen']) +
              cycler(linestyle=['o:', '+:', '*:', 'o-.','+_.','*-.','o--','+--','*--']))
    ax.set_prop_cycle(cycle3)

    for COI in costValue:

        xArrBudget,yArrBudget = findDeathForGivenCost(implementCostArrBudget,
                                        nDeathsBudget,
                                        tArrayBudget,
                                        costArrayBudget,
                                        COI)
        
        xArrR0,yArrR0 = findDeathForGivenCost(implementCostArrR0,
                                        nDeathsR0,
                                        tArrayR0,
                                        fracArrayR0,
                                        COI)

        xArrconstRvars,yArrconstRvars = findDeathForGivenCost(implementCostArrconstRvars,
                                        nDeathsconstRvars,
                                        tArrayconstRvars,
                                        fracArrayconstRvars,
                                        COI)

        USD = int(COI//1E9)


    
    ########################
    ### PLOTS
    ########################


        # Pick best option for given start date

        budgetIx = np.array(np.lexsort((yArrBudget,xArrBudget))).astype(int)
        xArrBudget = np.array(xArrBudget)
        yArrBudget = np.array(yArrBudget)
        xArrBudget = xArrBudget[budgetIx]
        yArrBudget = yArrBudget[budgetIx]
        [xArrBudget,budgetIx] = np.unique(xArrBudget,return_index = True)
        yArrBudget = yArrBudget[budgetIx.astype(int)]


        R0Ix = np.array(np.lexsort((yArrR0,xArrR0))).astype(int)
        xArrR0 = np.array(xArrR0)
        yArrR0 = np.array(yArrR0)
        xArrR0 = xArrR0[R0Ix]
        yArrR0 = yArrR0[R0Ix]
        [xArrR0,R0Ix] = np.unique(xArrR0,return_index = True)
        yArrR0 = yArrR0[R0Ix.astype(int)]

        constRvarsIx = np.array(np.lexsort((yArrconstRvars,xArrconstRvars))).astype(int)
        xArrconstRvars = np.array(xArrconstRvars)
        yArrconstRvars = np.array(yArrconstRvars)
        xArrconstRvars = xArrconstRvars[constRvarsIx]
        yArrconstRvars = yArrconstRvars[constRvarsIx]
        [xArrconstRvars,constRvarsIx] = np.unique(xArrconstRvars,return_index = True)
        yArrconstRvars = yArrconstRvars[constRvarsIx.astype(int)]


        # Do plots
        # Set color and linestyle cycle
        
        ax.plot(xArrBudget,yArrBudget,'o:',label='Daily budget w/ \${}bn tot. cost'.format(USD))
        ax.plot(xArrR0,yArrR0,'+:',label='$R_e$ fraction w/ \${}bn tot. cost'.format(USD))
        ax.plot(xArrconstRvars,yArrconstRvars,'*:',label='Target $R_e$  w/ \${}bn tot. cost'.format(USD))

    ax.set_xlabel('Control policy start day')
    ax.set_ylabel('Number of deaths')
    xLimits = dictVar['xLimits']
    yLimits = dictVar['yLimits']
    if len(xLimits)==2:
        ax.set_xlim(xLimits[0],xLimits[1])
    if len(yLimits)==2:
        ax.set_ylim(yLimits[0],yLimits[1])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    #fig.tight_layout()
    ax.set_title('Effect of control start date on total deaths')
    plt.show()

#################################

def interpolation_for_ParetoHeatmaps(rowArray,colArray,trueDeath , arr):
    '''
    Return:
        2D Array with the interpolated values 
    Purpose:
        [1]: Sort the columns of the [twoDArray] ( Ascending order)
        [2]: Interpolate [twoDArray] to a regular grid
    '''

    # Create the grid to interpolate to   
    nCol = len(colArray)
    nRow = len(rowArray)

    newRow = np.repeat( rowArray, nCol )
    col = np.tile( colArray , nRow )

    # re-sort columns (death index) for different parameter matrices
    idx = np.argsort(trueDeath)
    origRow = np.repeat( trueDeath[idx], nCol )
    
    
    # Do all the interpolations
    for qq in range(len(arr)):
        z = arr[qq][idx,: ].flatten()
        tmp = griddata( points = (origRow,col), 
                     values = z , 
                     xi = (newRow , col),
                     fill_value= 0,
                     method = 'linear')
        arr[qq]=np.reshape(tmp,(nRow,nCol))




    return arr




#################################

def ctrlMeasures_paretoHeatmaps_R(dictVar):

    recomputeData = dictVar['recomputeData']

    deathLevels = np.arange(5000, 120001 , 5000)

    tf = dictVar['tf']
    uMax = dictVar['uMax']

    if recomputeData:
        filePath = dictVar['filename'][0]
    
        with open(filePath,'r') as f:
            data = json.load(f)
        
        tArrayRtgt = np.array( data['tArray'] )
        fracArrayRtgt = np.array( data['fracArray'] )
        ACSArrRtgt = np.array(data['data'])
        implementCostArrRtgt = ACSArrRtgt [:,:,0] + ACSArrRtgt [:,:,1] + ACSArrRtgt [:,:,2] + ACSArrRtgt [:,:,3]
        nDeathsRtgt = ACSArrRtgt[:,:,6]
    
        # Smoothing
        tmp = 1.*implementCostArrRtgt
        implementCostArrRtgt[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                    tmp[:-2,2:] + tmp[1:-1,:-2] +
                                    tmp[1:-1,2:] + tmp[2:,:-2] +
                                    tmp[2:,1:-1] + tmp[2:,2:] +
                                    4*tmp[1:-1,1:-1])/12                                
        tmp = 1.*nDeathsRtgt
        nDeathsRtgt[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                    tmp[:-2,2:] + tmp[1:-1,:-2] +
                                    tmp[1:-1,2:] + tmp[2:,:-2] +
                                    tmp[2:,1:-1] + tmp[2:,2:] +
                                    4*tmp[1:-1,1:-1])/12                                
    
        startDateArr,fracArr,_,_ = findMinCostForGivenDeath(implementCostArrRtgt,
                                                nDeathsRtgt,
                                                tArrayRtgt,
                                                fracArrayRtgt,
                                                deathLevels)
    
        nSize = len(startDateArr)
    
        uLowTest = np.zeros( (nSize,tf) )
        uHighTest = np.zeros( (nSize,tf) )
        uLowDist = np.zeros( (nSize,tf) )
        uHighDist = np.zeros( (nSize,tf) )
    
        xCumDeaths = np.zeros((nSize,tf) )
        xRecovered = np.zeros((nSize,tf))
        xInfected = np.zeros((nSize,tf))
        xHosp = np.zeros((nSize,tf))
        
        for ii in range(nSize):
            print('RUN: {} out of {}'.format(ii+1,nSize))
    
            startDate = startDateArr[ii]
            frac = fracArr[ii]
    
            xTot, _ , uCtrlArr = optimalMinCost_R(dictVar, startDate, frac) 
    
            uLowTest[ii] = uCtrlArr[0]
            uHighTest[ii] = uCtrlArr[1]
            uLowDist[ii] = uCtrlArr[2]
            uHighDist[ii] = uCtrlArr[3] 
            
            xCumDeaths[ii] = xTot[:,8] + xTot[:,17] 
            xRecovered[ii] = xTot[:,7] + xTot[:,16]
            xInfected[ii] =  np.sum(xTot[:,1:7],axis=1)+ np.sum(xTot[:,10:16],axis=1)
            xHosp[ii] = xTot[:,6]+xTot[:,15]
            
        ## Smoothing
        uLowTest[:,1:-1] = (uLowTest[:,0:-2]+uLowTest[:,1:-1]+uLowTest[:,2:])/3
        uHighTest[:,1:-1] = (uHighTest[:,0:-2]+uHighTest[:,1:-1]+uHighTest[:,2:])/3
        uLowDist[:,1:-1] = (uLowDist[:,0:-2]+uLowDist[:,1:-1]+uLowDist[:,2:])/3
        uHighDist[:,1:-1] = (uHighDist[:,0:-2]+uHighDist[:,1:-1]+uHighDist[:,2:])/3    

        # Write data
        data = {
            'uLowTest': uLowTest.tolist(),
            'uHighTest': uHighTest.tolist(),
            'uLowDist': uLowDist.tolist(),
            'uHighDist': uHighDist.tolist(),
            'xCumDeaths': xCumDeaths.tolist(),
            'xRecovered': xRecovered.tolist(),
            'xInfected': xInfected.tolist(),
            'xHosp': xHosp.tolist() }
    
        filePath = dictVar['filename'][1]
    
        with open(filePath,'w') as f:
            json.dump(data, f,indent = 4)

    else: # Read data from file


        filePath = dictVar['filename'][1]
    
        with open(filePath,'r') as f:
            data = json.load(f)

        uLowTest = np.array(data['uLowTest'])
        uHighTest = np.array(data['uHighTest'])
        uLowDist = np.array(data['uLowDist'])
        uHighDist = np.array(data['uHighDist'])
        xCumDeaths = np.array(data['xCumDeaths'])
        xRecovered = np.array(data['xRecovered'])
        xInfected = np.array(data['xInfected'])
        xHosp = np.array(data['xHosp'])

    

    rowArray = deathLevels
    colArray = list(range(180))
    trueDeaths = xCumDeaths[:,-1]
    limits = [0 , tf , min(trueDeaths) , max(trueDeaths) ]

    arr = [uLowTest, uHighTest , uLowDist , uHighDist , xCumDeaths,xRecovered, xInfected, xHosp]
 
    arrSorted = interpolation_for_ParetoHeatmaps(rowArray,colArray,trueDeaths, arr)
    
    uLowTestInterpol = arrSorted[0]
    uHighTestInterpol = arrSorted[1]
    uLowDistInterpol = arrSorted[2]
    uHighDistInterpol = arrSorted[3]

    xCumDeathsInterpol = arrSorted[4]
    xRecoveredInterpol = arrSorted[5]
    xInfectedInterpol= arrSorted[6]
    xHospInterpol = arrSorted[7]


    plt.figure(1)
    plt.imshow(uLowTestInterpol,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[0])
    plt.colorbar()
    plt.title('Low-Risk Testing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()


    plt.figure(2)
    plt.imshow(uHighTestInterpol,origin = 'lower',extent = limits,aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[1])
    plt.colorbar()
    plt.title('High-Risk Testing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()


    plt.figure(3)
    plt.imshow(uLowDistInterpol,origin = 'lower',extent = limits,aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[2])
    plt.colorbar()
    plt.title('Low-Risk Social Distancing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()



    plt.figure(4)
    plt.imshow(uHighDistInterpol,origin = 'lower',extent = limits,aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[3])
    plt.colorbar()
    plt.title('High-Risk Social Distancing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()


    plt.figure(5)
    plt.imshow(xCumDeathsInterpol,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Cumulative deaths')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()

    plt.figure(6)
    plt.imshow(xRecoveredInterpol,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Cumulative recovered')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()


    plt.figure(7)
    plt.imshow(xInfectedInterpol,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Current infected')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()


    plt.figure(8)
    plt.imshow(xHospInterpol,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Current hospitalized')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()


    plt.show()




def ctrlMeasures_paretoHeatmaps_Budget(dictVar):

    recomputeData = dictVar['recomputeData']


    deathLevels = np.arange(0, 120001 , 5000)


    tf = dictVar['tf']
    uMax = dictVar['uMax']

    if recomputeData:
        filePath = dictVar['filename'][0]
    
        with open(filePath,'r') as f:
            data = json.load(f)
    
        tArrayBudget = np.array( data['tArray'] )
        costArrayBudget = np.array( data['costArray'] )
        ACSArrBudget = np.array(data['data'])   
        implementCostArrBudget = ACSArrBudget[:,:,0] + ACSArrBudget[:,:,1] + ACSArrBudget[:,:,2] + ACSArrBudget[:,:,3]
        nDeathsBudget = ACSArrBudget[:,:,6]
    
    
        ####################
        # Smoothing
        tmp = 1.*implementCostArrBudget
        implementCostArrBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                    tmp[:-2,2:] + tmp[1:-1,:-2] +
                                    tmp[1:-1,2:] + tmp[2:,:-2] +
                                    tmp[2:,1:-1] + tmp[2:,2:] +
                                    4*tmp[1:-1,1:-1])/12                                
        tmp = 1.*nDeathsBudget
        nDeathsBudget[1:-1,1:-1] =  (tmp[:-2,:-2] + tmp[:-2,1:-1] +
                                    tmp[:-2,2:] + tmp[1:-1,:-2] +
                                    tmp[1:-1,2:] + tmp[2:,:-2] +
                                    tmp[2:,1:-1] + tmp[2:,2:] +
                                    4*tmp[1:-1,1:-1])/12                                 
    
        startDateArr,costArr,_,_ = findMinCostForGivenDeath(implementCostArrBudget,
                                                nDeathsBudget,
                                                tArrayBudget,
                                                costArrayBudget,
                                                deathLevels)
    
        nSize = len(startDateArr)
    
        uLowTest = np.zeros( (nSize,tf) )
        uHighTest = np.zeros( (nSize,tf) )
        uLowDist = np.zeros( (nSize,tf) )
        uHighDist = np.zeros( (nSize,tf) )
    
        xCumDeaths = np.zeros((nSize,tf) )
        xRecovered = np.zeros((nSize,tf))
        xInfected = np.zeros((nSize,tf))
        xHosp = np.zeros((nSize,tf))
    
        for ii in range(nSize):
            print('RUN: {} out of {}'.format(ii+1,nSize))
    
            startDate = startDateArr[ii]
            cost = costArr[ii]
    
            xTot, _ , uCtrlArr = optimalMinCost_Budget(dictVar, startDate, cost) 
    
            uLowTest[ii] = uCtrlArr[0]
            uHighTest[ii] = uCtrlArr[1]
            uLowDist[ii] = uCtrlArr[2]
            uHighDist[ii] = uCtrlArr[3]
    
            xCumDeaths[ii] = xTot[:,8] + xTot[:,17] 
            xRecovered[ii] = xTot[:,7] + xTot[:,16]
            xInfected[ii] =  np.sum(xTot[:,1:7],axis=1)+ np.sum(xTot[:,10:16],axis=1)
            xHosp[ii] = xTot[:,6]+xTot[:,15]

        # Write data
        data = {
            'uLowTest': uLowTest.tolist(),
            'uHighTest': uHighTest.tolist(),
            'uLowDist': uLowDist.tolist(),
            'uHighDist': uHighDist.tolist(),
            'xCumDeaths': xCumDeaths.tolist(),
            'xRecovered': xRecovered.tolist(),
            'xInfected': xInfected.tolist(),
            'xHosp': xHosp.tolist() }
    
        filePath = dictVar['filename'][1]
    
        with open(filePath,'w') as f:
            json.dump(data, f,indent = 4)
    
    else: # Read data from file
        filePath = dictVar['filename'][1]
    
        with open(filePath,'r') as f:
            data = json.load(f)
        uLowTest = np.array(data['uLowTest'])
        uHighTest = np.array(data['uHighTest'])
        uLowDist = np.array(data['uLowDist'])
        uHighDist = np.array(data['uHighDist'])
        xCumDeaths = np.array(data['xCumDeaths'])
        xRecovered = np.array(data['xRecovered'])
        xInfected = np.array(data['xInfected'])
        xHosp = np.array(data['xHosp'])


    ## Plot data
    
    
    limits = [0,tf,min(deathLevels),max(deathLevels)]
    
    plt.figure(1)
    plt.imshow(uLowTest,origin = 'lower',extent = limits,aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[1])
    plt.colorbar()
    plt.title('Low-Risk Testing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')
   

    plt.figure(2)
    plt.imshow(uHighTest,origin = 'lower',extent = limits,aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[1])
    plt.colorbar()
    plt.title('High-Risk Testing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')

    plt.figure(3)
    plt.imshow(uLowDist,origin = 'lower',extent = limits,aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[2])
    plt.colorbar()
    plt.title('Low-Risk Social Distancing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')

    plt.figure(4)
    plt.imshow(uHighDist,origin = 'lower',extent = limits,aspect='auto',interpolation = 'gaussian',vmin=0, vmax=uMax[3])
    plt.colorbar()
    plt.title('High-Risk Social Distancing')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')

    plt.figure(5)
    plt.imshow(xCumDeaths,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Cumulative deaths')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')

    plt.figure(6)
    plt.imshow(xRecovered,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Cumulative recovered')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')

    plt.figure(7)
    plt.imshow(xInfected,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Current infected')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')

    plt.figure(8)
    plt.imshow(xHosp,origin = 'lower',extent = limits, aspect='auto',interpolation = 'gaussian')
    plt.colorbar()
    plt.title('Current hospitalized')
    plt.xlabel('Day')
    plt.ylabel('Number of Deaths')



    plt.show()





