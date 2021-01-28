# Different parameter values
# 1. increase beta by 25%  beta = "Baseline transmission rate"
# 2. decrease tau by 25% "Symptomatic proportion" (tau) 
# 3. increase wA by 25% "rel. infect. of symptomatic" 
# 4. increase "exposed rate" sigma by 25%
# 5. increase all testing cost coefficients by 25%
# 6. reduce  all distancing cost coefficients by 25%


#################
## CHECK THE END OF THE FILE FOR THE PLOTS OF THIS FILE
################

import  numpy as np
from scipy.sparse.linalg import eigs as speigs
import matplotlib.pyplot as plt

import scipy.optimize as opt 
from skimage import measure


def initialization():

    # Initial conditions
    N = [1340000,423000]
    NA = [1340000,423000]
    xI = np.zeros(18)
    xI[0] = N[0] # Susceptible low risk
    xI[9] = N[1] # Susceptible high risk
    xI[1] = 10000
    xI[10] = 0

    # System parameters
    beta = 0.0640
    Phi = np.array([[10.56,2.77],[9.4,2.63]]) ###Contact matrix

    gamA = 1/4####1/gamA~4
    gamY = 1/4###gamA=gamY##1/gamA~4
    gamH = 1/10.7
    eta = 0.1695
    tau = 0.57
    sig = 1/2.9###1/sig~2.9
    rhoA = 1/2.3###rhoA=rhoY##1/rhoY~2.3
    rhoY = 1/2.3###1/rhoY~2.3
    P = 0.44#from paper
    wY =1.0#from paper
    wA =0.66#from paper

    IFR = np.array([0.6440/100,6.440/100])
    YFR = np.array([IFR[0]/tau,IFR[1]/tau]) 
    YHR = np.array([4.879/100,48.79/100])
    HFR = np.array([YFR[0]/YHR[0],YFR[1]/YHR[1]])

    wP= P/(1-P) /(tau*wY/rhoY + (1-tau)*wA/rhoA) \
        * ((1-tau)*wA/gamA \
        + tau*wY* np.array([YHR[0]/eta+(1-YHR[0])/gamY, \
                        YHR[1]/eta+(1-YHR[1])/gamY])) 
            
    wPY = wP*wY# from paper
    wPA = wP*wA #from paper

    Pi = gamY*np.array([YHR[0]/(eta+(gamY-eta)*YHR[0]),\
                    YHR[1]/(eta+(gamY-eta)*YHR[1])])# Array with two values

    mu = 1/8.1###1/mu~8.1
    theta = 3000 #2352 ventilators in Houston (https://www.click2houston.com/health/2020/04/10/texas-medical-center-data-shows-icu-ventilator-capacity-vs-usage-during-coronavirus-outbreak/)
    nu = gamH*np.array([HFR[0]/(mu+(gamH-mu)*HFR[0]),\
                    HFR[1]/(mu+(gamH-mu)*HFR[1])])# Array with two values


    a = np.array([[0,2.3,27],[0,2.3,27]]) # Testing costs
    b = np.array([[0,0,40],[0,0,40]]) # Distancing costs


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
        'b':b }

    return dictVar



def compCost(u,dictVar,imm):
    '''
    Parameters:
        dictVar: Dictionary containing the variables 
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

    # COST FOR SOCIAL DISTANCING AND TESTING 
    totalCost = 0 


    #######################
    ### COMPUTE TOTAL COST
    #######################

    S = (1-imm)*np.array(N)
    NA = 1.* S

    # Costs
    # Costs
    # Constant, linear and quadratic testing costs
    # for low risk and high risk
    # First cost -- low risk testing cost
    # Second cost -- high risk testing cost
    # $5 test, 97% accurate
    # https://www.sciencemag.org/news/2020/08/milestone-fda-oks-simple-accurate-coronavirus-test-could-cost-just-5 
    
    # Constant, linear, quadratic Distancing costs
    # For low risk and high risk
    # Third cost -- low risk distancing cost
    # Fourth cost -- high risk distancing cost


    # SOCIAL DISTANCE (LOW RISK)  
    distCost = (u[0]>0)*a[0,0]+a[0,1]*NA[0]*u[0]+a[0,2]*NA[0]*u[0]**2
    # SOCIAL DISTANCE (HIGH RISK)
    distCost = distCost+(u[1]>0)*a[1,0]+a[1,1]*NA[1]*u[1]+a[1,2]*NA[1]*u[1]**2


    # TESTING (LOW RISK)
    testCost =  b[0,0]*N[0]+b[0,1]*N[0]*u[2]+b[0,2]*N[0]*u[2]**2
    # TESTING (HIGH RISK) 
    testCost = testCost + b[1,0]*N[1]+b[1,1]*N[1]*u[3]+b[1,2]*N[1]*u[3]**2

    totalCost = distCost + testCost


    # print('u:',u)
    # print('imm:',imm)
    # print('cost:',totalCost)
    # print('')


    return totalCost



def compReproductionNumber(u,dictVar,Imm):
    '''
    Parameters:
        u: Control rates
            U[0]: Low risk testing rate ( 0-1 )
            U[1]: High risk resting rate ( 0-1 )
            U[2]: Low risk distance rate ( 0-1 )
            U[3]: High risk distance rate ( 0-1 )
        dictVar: Dictionary containing the variables 
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


    # POPULATION THAT IS NOT IMMUNE 
    S = (1-Imm)*np.array(N)
    NA = 1.* S

    ###############################
    ### COMPUTE REPRODUCTION NUMBER
    ###############################

    F00 = np.array(\
            [np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])*(1-u[2])*beta*S[0]/N[0]*Phi[0,0],
            [(1-tau)*sig,0,0,0,0],
            [tau*sig,0,0,0,0],
            [0,rhoA,0,0,0],
            [0,0,rhoY,0,0]])
    
    F11 = np.array(\
            [np.array([0,(1-u[1])*wPA[0],(1-u[1])*wPY[0],(1-u[1])*wA,wY])\
                *(1-u[3])*beta*S[1]/N[1]*Phi[1,1],
            [(1-tau)*sig,0,0,0,0],
            [tau*sig,0,0,0,0],
            [0,rhoA,0,0,0],
            [0,0,rhoY,0,0]])
    
    F01 = np.array(\
            [np.array([0,(1-u[1])*wPA[1],(1-u[1])**wPY[1],(1-u[1])*wA,wY])\
                *(1-u[2])*beta*S[0]/N[1]*Phi[0,1],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]])
    
    F10 = np.array(\
            [np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])\
                *(1-u[3])*beta*S[1]/N[0]*Phi[1,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]])
    
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
    

    # FOR SOME ODD REASON, I SAM GETTING NAN FROM COMPUTATIONS... SO REPLACING WITH NAN WITH 0
    F00 = np.nan_to_num(F00, nan = 0)
    F11 = np.nan_to_num(F11, nan = 0)
    F01 = np.nan_to_num(F01, nan = 0)
    F10 = np.nan_to_num(F10, nan = 0)
    V00 = np.nan_to_num(V00, nan = 0)
    V11 = np.nan_to_num(V11, nan = 0)
    

    V10 = np.zeros((5,5))
    V01 = np.copy(V10)
    
    F = np.bmat([[F00,F01],[F10,F11]])
    V = np.bmat([[V00,V01],[V10,V11]])


    try:
        
        V_inv = np.linalg.inv(V)
        Prod = F*V_inv

        val,__ = speigs(Prod,k=1)

        # REPRODUCTION NUMBER 
        rho = np.real(val[0])

        #print('Rho:',rho)
        return rho


    except Exception as e:        
        
        print('\n--------- MESSAGE ---------')
        print(e)
        print('** Will Try to run again. **')


    try:

        # TRY TO RUN IT AGAIN
        V_inv = np.linalg.inv(V)
        Prod = F*V_inv

        val,__ = speigs(Prod,k=1)

        # REPRODUCTION NUMBER 
        rho = np.real(val[0])

        return rho

    except Exception as e:

        # STATE THE PARAMETERS THAT CAUSED THE ERROR
        print('\n--------- ERROR ---------')
        print('Function: compReproductionNumber')
        print('uVec:',u)
        print('immunity:',Imm)


        #print( arr * (1-u[3])*beta*S[0] / N[1]*Phi[0,1])

        # PRINT ERROR MESSAGE
        print('\n--------- MESSAGE ---------')
        print(e)
        print('-------------------------\n')




def contourPlots(dictVar,t0,tf,tStep,d0,df,dStep,imm):
    '''
    Parameter:
        t0: Minimum testing control rate
        tf: Maximum testing control rate
        tStep: Step size between t0 and tf
        d0: Minimum social distance control rate
        df:  Maximum social distance control rate
        dStep: Step size between d0 and df
    '''

    # HIGH/LOW TEST RATE ARRAY
    tRateArray = np.arange(0,tf + tStep,tStep)

    # HIGH/LOW DISTANCE RATE ARRAY
    dRateArray = np.arange(0,df + dStep,dStep)

    # SIZE OF THE HIGH/LOW TEST RATE ARRAY
    nTRA = len(tRateArray)

    # SIZE OF THE HIGH/LOW DISTANCE RATE ARRAY
    nDRA = len(dRateArray)

    # REPRODUCTION NUMBER
    rhoVal = np.zeros((nTRA, nDRA))

    # COST
    costVal = np.zeros((nTRA,nDRA)) 

    # [0,0,0,0] * CURRENT CONTROL RATE  
    u = np.zeros(4)    

    for i,tRate in enumerate(tRateArray):
        for j, dRate in enumerate(dRateArray):

            # U[0]: LOW RISK TESTING RATE
            # U[1]: HIGH RISK TESTING RATE
            # U[2]: LOW RISK DISTANCING RATE
            # U[3]: HIGH RISK DISTANCING RATE
            u[0] = tRate
            u[1] = tRate
            u[2] = dRate
            u[3] = dRate
        

            costVal[i,j] = compCost(u,dictVar,imm)
            rhoVal[i,j] = compReproductionNumber(u,dictVar,imm)
                


    ################################ 
    
    fig1,ax1 = plt.subplots()    
    pos1 = ax1.contourf(dRateArray,
                        tRateArray,
                        costVal[:,:], levels=20) 

    cbar1 = fig1.colorbar(pos1, ax=ax1) 
    cbar1.set_label('Cost ($10M USD/day)')
    
    CS1 = ax1.contour(dRateArray,
                    tRateArray,
                    rhoVal[:,:],
                    colors = 'w')

    blackDots =  [  [], [] ]
    fmt1 = {}
    for j,R0 in enumerate(CS1.levels):

        if len(CS1.allsegs[j]) > 0:


            fmt1[R0] = "$R_e$={:0.2f}".format(R0)

            # FIND THE CONTOUR POINTS FOR THE GIVEN DEATH COST
            contours = np.array(measure.find_contours(rhoVal, R0,'low'))[0]
            
            # Interpolation routine
            tmp = np.zeros(len(contours))            
            # Loop through the arrays 
            for k, xy in enumerate(contours):
               [xx,yy] = np.floor(xy).astype(int)
               xx = min([xx,len(tRateArray)-2])
               yy = min([yy,len(dRateArray)-2])
               
               tmpMx = np.array([[1,xx,yy,xx*yy],
                                [1,xx+1,yy,(xx+1)*yy],
                                [1,xx,yy+1,xx*(yy+1)],
                                [1,xx+1,yy+1,(xx+1)*(yy+1)]])
               tmpVec = np.array([costVal[xx,yy],
                               costVal[xx+1,yy],
                               costVal[xx,yy+1],
                               costVal[xx+1,yy+1]])
               coeffs = np.linalg.solve(tmpMx,tmpVec)
               tmp[k] = coeffs[0] + coeffs[1]*xy[0] + coeffs[2]*xy[1] \
                        + coeffs[3]*xy[0]*xy[1]
            
            # FIND THE SMALLEST COST
            minIdx = np.argmin(tmp)

            # FIND THE INCEDES FROM THE CONTOUR POINTS THAT HAD THE SMALLEST COST
            # THIS IS MORE OF AN ESTIMATE THAN AN EXACT FACT
            curIdx = contours[minIdx]

            # MAP THE INDEX TO THE X,Y AXIS VALUE
            tPoint = curIdx[0] *tStep + tRateArray[0]
            cPoint = curIdx[1] *tStep + dRateArray[0]

            # STORE TEH POINTS 
            blackDots[0].append( tPoint )
            blackDots[1].append( cPoint )


        else:
            fmt1[R0] = ''
            
    # # Label every other level using strings
    ax1.clabel(CS1,
            CS1.levels,
            inline=True,
            fmt=fmt1,
            fontsize=10)


    ax1.scatter(blackDots[1],blackDots[0],marker = '*',s = 100 ,color = 'k',zorder=2)

    ax1.set_title('Immunity rate:{}'.format(imm),fontsize = 14)
    ax1.set_xlabel('Social distance control level',fontsize = 14)
    ax1.set_ylabel('Testing control level',fontsize = 14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
        

    plt.show()


if __name__ == '__main__':


    dictVar = initialization()

    dictVar['uMax'] = np.array([0.666,0.666,0.8,0.8])

    contourPlots(dictVar,t0 = 0, tf = 0.666, tStep = 0.02,
                        d0 = 0, df = 0.8, dStep = 0.02, imm = 0.0)

    contourPlots(dictVar,t0 = 0, tf = 0.666, tStep = 0.02,
                        d0 = 0, df = 0.8, dStep = 0.02, imm = 0.666)                        









