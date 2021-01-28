
# distance_testing_param_contour() 
# 1. +/- beta by 25%  beta = "Baseline transmission rate"
# 2. +/- tau by 25% "Symptomatic proportion" (tau) 
# 3. +/- wA by 25% "rel. infect. of asymptomatic" 

# distance_testing_multiplier_contour
# 4. +/- all testing cost coefficients by 25%
# 5. +/-  all distancing cost coefficients by 25%

#################
## CHECK THE END OF THE FILE FOR THE PLOTS OF THIS FILE
################

import numpy as np
from scipy.sparse.linalg import eigs as speigs
import matplotlib.pyplot as plt



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
    beta =0.0640
    Phi = np.array([[10.56,2.77],
                    [9.4,2.63]]) ###Contact matrix
    
    gamA = 1/4####1/gamA~4
    gamY = 1/4###gamA=gamY##1/gamA~4
    gamH = 1/10.7
    eta = 0.1695
    tau = 0.57 # asymptomatic proprotion
    sig = 1/2.9### exposed rate
    rhoA = 1/2.3###rhoA=rhoY##1/rhoY~2.3
    rhoY = 1/2.3###1/rhoY~2.3
    P = 0.44
    wY =1.0
    wA =0.66
    
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
    
    mu = 1/8.1###1/mu~8.1
    theta = 3000 #2352 ventilators in Houston (https://www.click2houston.com/health/2020/04/10/texas-medical-center-data-shows-icu-ventilator-capacity-vs-usage-during-coronavirus-outbreak/)
    nu = gamH*np.array([HFR[0]/(mu+(gamH-mu)*HFR[0]),\
                      HFR[1]/(mu+(gamH-mu)*HFR[1])])# Array with two values    
        
        
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

    a = np.array([[0,2.3,27],[0,2.3,27]]) # Testing costs
    b = np.array([[0,0,40],[0,0,40]]) # Distancing costs

    c = [100,100]     # Cost 5: Opportunity cost for sickness per day (low and high risk)
    d =  [500,750]   # Cost 6: Hospitalization cost per day  (low and high risk)
    e = [100000,75000] # Death
    f = [5000,5000] # Cost of remaining infected
    tf = 120           # Final time
    nsubDiv = 10 # Number of subdivisions per interval
    # (0:n)*tf/(n-1)
    
    # Default control values
    u1max = 0.66# high-risk testing
    u0max = 1*u1max# About 50 percent accuracy
    # $5 test, 97% accurate
    # https://www.sciencemag.org/news/2020/08/milestone-fda-oks-simple-accurate-coronavirus-test-could-cost-just-5 
    
    v1max = 0.8 # Maximum level of distancing for high risk
    v0max = 1*v1max # Max level of distancing for low risk
    frac0 = 0.3# Low level of control for low risk (as fraction of max)
    frac1 = 0.3# Low level of ocntrol for high risk (as frac. of max)
    
    
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
            'nsubDiv':nsubDiv,
            'u1max':u1max,
            'u0max':u0max,
            'v1max':v1max,
            'v0max':v0max,
            'frac0':frac0,
            'frac1':frac1}



    return dictVar



# CREATE CONTOUR PLOTS FOR TESTING VS SOCIAL DISTANCE
def distance_testing_contour():
    
    dictVar = initialization()
    
    # INITIAL CONDITION
    
    N = dictVar['N']
    NA = dictVar['NA']
    xI = dictVar['xI']
    
    
    # SYSTEM PARAMETERS
    
    beta = dictVar['beta']
    Phi = dictVar['Phi']
    gamA = dictVar['gamA']
    gamY = dictVar['gamY']
    gamH = dictVar['gamH']
    eta = dictVar['eta']
    tau = dictVar['tau']
    sig = dictVar['sig']
    rhoA = dictVar['rhoA']
    rhoY = dictVar['rhoY']
    P = dictVar['P']
    wY = dictVar['wY']
    wA = dictVar['wA']
    IFR = dictVar['IFR']
    YFR = dictVar['YFR']
    YHR = dictVar['YHR']
    HFR = dictVar['HFR']
    wP = dictVar['wP']
    wPY = dictVar['wPY']
    wPA = dictVar['wPA']
    Pi = dictVar['Pi']
    mu = dictVar['mu']
    theta = dictVar['theta']
    nu = dictVar['nu']

        

    Imm = [0.0, 0.666] # IMMUNITY RATE OF INDIVIDUALS

    

    tStep = 0.025
    dStep = 0.025


    a = np.array([[0,2.3,27],[0,2.3,27]]) # Testing costs
    b = np.array([[0,0,40],[0,0,40]]) # Distancing costs
    u1max = 0.66
    v1max = 0.8


    # HIGH/LOW TEST RATE ARRAY
    tRateArray = np.arange(0,u1max,tStep)

    # HIGH/LOW DISTANCE RATE ARRAY
    dRateArray = np.arange(0,v1max,dStep)

    # SIZE OF THE HIGH/LOW TEST RATE ARRAY
    nTRA = len(tRateArray)

    # SIZE OF THE HIGH/LOW DISTANCE RATE ARRAY
    nDRA = len(dRateArray)



    rhoVal = np.zeros((nTRA, nDRA,len(Imm))) # REPRODUCTION NUMBER
    costVal = np.zeros((nTRA,nDRA,len(Imm))) # COST

    u = np.zeros(4)    # [0,0,0,0] * CURRENT CONTROL RATE


    for j, dRate in enumerate(dRateArray):
        for i,tRate in enumerate(tRateArray):
        
            # U[0]: LOW RISK TESTING RATE
            # U[1]: HIGH RISK TESTING RATE
            # U[2]: LOW RISK DISTANCING RATE
            # U[3]: HIGH RISK DISTANCING RATE
            u[0] = tRate
            u[1] = tRate
            u[2] = dRate
            u[3] = dRate
        
            for k in range(len(Imm)): # IMMUNITY RATE
                
                S = (1-Imm[k])*np.array(N)
                NA = 1.*S
 
    
                # Cost for testing (low risk)
                costTmp = (u[0]>0)*a[0,0]+a[0,1]*NA[0]*u[0]+a[0,2]*NA[0]*u[0]**2
                # Cost for testing (high risk)
                costTmp = costTmp+(u[1]>0)*a[1,0]+a[1,1]*NA[1]*u[1]+a[1,2]*NA[1]*u[1]**2
                # Cost for social distance (low risk)
                costTmp = costTmp + b[0,0]*N[0]+b[0,1]*N[0]*u[2]+b[0,2]*N[0]*u[2]**2
                # Cost for social distance (high risk)
                costTmp = costTmp + b[1,0]*N[1]+b[1,1]*N[1]*u[3]+b[1,2]*N[1]*u[3]**2
                costVal[i,j,k] = costTmp
               
          
                F00 = np.array(\
                      [np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])\
                             *(1-u[2])*beta*S[0]/N[0]*Phi[0,0],
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
                
                F = np.bmat([[F00,F01],[F10,F11]])
                V = np.bmat([[V00,V01],[V10,V11]])
                V_inv = np.linalg.inv(V)
                Prod = F*V_inv
                val,__ = speigs(Prod,k=1)
                rhoVal[i,j,k]=np.real(val[0])
        

    ################################ 
    fig1,ax1 = plt.subplots()    
    pos1 = ax1.contourf(costVal[:,:,0],
                    extent = [0,u1max,0,v1max]) 

    cbar1 = fig1.colorbar(pos1, ax=ax1) 
    cbar1.set_label('Cost (US Dollars)')
    
    CS1 = ax1.contour(rhoVal[:,:,0],
                    colors = 'w',
                    extent = [0,u1max,0,v1max])

    fmt1 = {}

    # MAKE SURE TO LABEL ONLY THE CONTOUR LINES AND NOT THE EMPTY
    for j,lvl in enumerate(CS1.levels):
        if len(CS1.allsegs[j]) > 0:
            # print(CS1.allsegs[i])
            fmt1[lvl] = "R$_0$={:0.2f}".format(lvl)
            i-=1
        else:
            fmt1[lvl] = ''
            
    ax1.clabel(CS1,
            CS1.levels,
            inline=True,
            fmt=fmt1,
            fontsize=10)

    ax1.set_title('Immunity rate:{}'.format(Imm[0]),fontsize = 14)
    ax1.set_xlabel('Social distance control level',fontsize = 14)
    ax1.set_ylabel('Testing control level',fontsize = 14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
        
    ################################
    fig2,ax2 = plt.subplots()    
    pos2 = ax2.contourf(costVal[:,:,1],
                    extent = [0,u1max,0,v1max],levels=15) 

    cbar2 = fig2.colorbar(pos2, ax=ax2) 
    cbar2.set_label('Cost (US Dollars/day)')

    CS2 = ax2.contour(rhoVal[:,:,1],
                    colors = 'w',
                    extent = [0,u1max,0,v1max])


    fmt2 = {}
    for j,lvl in enumerate(CS2.levels):

        if len(CS2.allsegs[j]) > 0:
            # print(CS1.allsegs[i])
            fmt2[lvl] = "R$_e$={:0.2f}".format(lvl)
            i-=1
        else:
            fmt1[lvl] = ''

    ax2.clabel(CS2, 
            CS2.levels, 
            inline=True, 
            fmt=fmt2, 
            fontsize=10)

    ax2.set_title('Immunity rate:{}'.format(Imm[1]),fontsize = 14) 
    ax2.set_xlabel('Social distance control level',fontsize = 14)
    ax2.set_ylabel('Testing control level',fontsize = 14)  
    ax2.tick_params(axis='both', which='major', labelsize=14) 

    ################################
    fig3,ax3 = plt.subplots()    
    pos3 = ax3.contourf(costVal[:,:,2],
                    extent = [0,u1max,0,v1max], levels=15) 
    
    cbar3 = fig3.colorbar(pos3, ax=ax3) 
    cbar3.set_label('Cost ($USD / day)')

    CS3 = ax3.contour(rhoVal[:,:,2],
                    colors = 'w',
                    extent = [0,u1max,0,v1max])


    fmt3 = {}
    for j,lvl in enumerate(CS3.levels):

        if len(CS3.allsegs[j]) > 0:
            # print(CS1.allsegs[i])
            fmt3[lvl] = "R$_0$={:0.2f}".format(lvl)
            i-=1
        else:
            fmt1[lvl] = ''

    ax3.clabel(CS3, 
            CS3.levels, 
            inline=True, 
            fmt=fmt3, 
            fontsize=10)


        
    ax3.set_title('Immunity rate:{}'.format(Imm[2]),fontsize = 14)
    ax3.set_xlabel('Social distance control level',fontsize = 14)
    ax3.set_ylabel('Testing control level',fontsize = 14)    
    ax3.tick_params(axis='both', which='major', labelsize=14)  
        

    plt.show() 
    
    
#############################################
### THESE METHODS DEPEND ON EACH OTHER ######

# CREATE PLOTS FOR TAU/WA/BETA

def getRhoVal(param,varArray,dictV,u,S):


    dictVar = dictV
    
    # INITIAL CONDITION
    
    N = dictVar['N']
    NA = dictVar['NA']
    xI = dictVar['xI']
    
    
    # SYSTEM PARAMETERS
    
    beta = dictVar['beta']
    Phi = dictVar['Phi']
    gamA = dictVar['gamA']
    gamY = dictVar['gamY']
    gamH = dictVar['gamH']
    eta = dictVar['eta']
    tau = dictVar['tau']
    sig = dictVar['sig']
    rhoA = dictVar['rhoA']
    rhoY = dictVar['rhoY']
    P = dictVar['P']
    wY = dictVar['wY']
    wA = dictVar['wA']
    IFR = dictVar['IFR']
    YFR = dictVar['YFR']
    YHR = dictVar['YHR']
    HFR = dictVar['HFR']
    wP = dictVar['wP']
    wPY = dictVar['wPY']
    wPA = dictVar['wPA']
    Pi = dictVar['Pi']
    mu = dictVar['mu']
    theta = dictVar['theta']
    nu = dictVar['nu']


    rhoVal = np.zeros( len(varArray) )

    for idx,x in enumerate(varArray):

        if param == 'beta':
            beta = x
        elif param == 'tau':
            tau = x
        elif param == 'wA':
            wA = x
        else:
            raise('wrong param')


        F00 = np.array([np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])*(1-u[2])*beta*S[0]/N[0]*Phi[0,0],
                [(1-tau)*sig,0,0,0,0],
                [tau*sig,0,0,0,0],
                [0,rhoA,0,0,0],
                [0,0,rhoY,0,0]])
        
        F11 = np.array([
                np.array([0,(1-u[1])*wPA[0],(1-u[1])*wPY[0],(1-u[1])*wA,wY])*(1-u[3])*beta*S[1]/N[1]*Phi[1,1],
                [(1-tau)*sig,0,0,0,0],
                [tau*sig,0,0,0,0],
                [0,rhoA,0,0,0],
                [0,0,rhoY,0,0]])
        
        F01 = np.array([
                np.array([0,(1-u[1])*wPA[1],(1-u[1])**wPY[1],(1-u[1])*wA,wY])*(1-u[3])*beta*S[0]/N[1]*Phi[0,1],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0]])
        
        F10 = np.array([
                np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])*(1-u[2])*beta*S[1]/N[0]*Phi[1,0],
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
        
        F = np.bmat([[F00,F01],[F10,F11]])
        V = np.bmat([[V00,V01],[V10,V11]])
        V_inv = np.linalg.inv(V)
        Prod = F*V_inv
        val,__ = speigs(Prod,k=1)
        rhoVal[idx] = np.real(val[0])   

    return rhoVal



def distance_testing_param_contour():
    
    dictVar = initialization()
    
    # INITIAL CONDITION
    
    N = dictVar['N']
    NA = dictVar['NA']
    xI = dictVar['xI']
    
    
    # SYSTEM PARAMETERS
    
    beta = dictVar['beta']
    Phi = dictVar['Phi']
    gamA = dictVar['gamA']
    gamY = dictVar['gamY']
    gamH = dictVar['gamH']
    eta = dictVar['eta']
    tau = dictVar['tau']
    sig = dictVar['sig']
    rhoA = dictVar['rhoA']
    rhoY = dictVar['rhoY']
    P = dictVar['P']
    wY = dictVar['wY']
    wA = dictVar['wA']
    IFR = dictVar['IFR']
    YFR = dictVar['YFR']
    YHR = dictVar['YHR']
    HFR = dictVar['HFR']
    wP = dictVar['wP']
    wPY = dictVar['wPY']
    wPA = dictVar['wPA']
    Pi = dictVar['Pi']
    mu = dictVar['mu']
    theta = dictVar['theta']
    nu = dictVar['nu']

        

    Imm = [0.0, 0.666] # IMMUNITY RATE OF INDIVIDUALS

    tStep = 0.025
    dStep = 0.025

    a = np.array([[0,2.3,27],[0,2.3,27]]) # Testing costs
    b = np.array([[0,0,40],[0,0,40]]) # Distancing costs
    u1max = 0.66# 
    v1max = 0.8# 

    # HIGH/LOW TEST RATE ARRAY
    tRateArray = np.arange(0,u1max,tStep)

    # HIGH/LOW DISTANCE RATE ARRAY
    dRateArray = np.arange(0,v1max,dStep)

    # SIZE OF THE HIGH/LOW TEST RATE ARRAY
    nTRA = len(tRateArray)

    # SIZE OF THE HIGH/LOW DISTANCE RATE ARRAY
    nDRA = len(dRateArray)

    rateArray = np.array([0.75,1.0,1.25])

    # CREATE A BETA ARRAY SUCH THAT [75% OF BETA, BETA, 125% OF BETA]
    betaArray = np.array([beta]*3) * rateArray

    # CREATE A TAU ARRAY SUCH THAT [75% OF BETA, BETA, 125% OF BETA]
    tauArray = np.array([tau]*3) * rateArray

    # CREATE A WA ARRAY SUCH THAT [75% OF BETA, BETA, 125% OF BETA]
    wAArray = np.array([wA]*3) * rateArray

    rhoValBeta = np.zeros( (nTRA, nDRA,len(Imm),len(betaArray) ) ) # REPRODUCTION NUMBER
    rhoValWA = np.zeros( (nTRA, nDRA,len(Imm),len(tauArray) ) ) # REPRODUCTION NUMBER
    rhoValTau = np.zeros( (nTRA, nDRA,len(Imm),len(wAArray) ) ) # REPRODUCTION NUMBER

    costVal = np.zeros( (nTRA,nDRA,len(Imm) ) ) # TOTAL COST

    distCost = np.zeros( (nTRA,nDRA,len(Imm) ) )
    testCost = np.zeros( (nTRA,nDRA,len(Imm) ) )


    u = np.zeros(4)    # [0,0,0,0] * CURRENT CONTROL RATE


    for j, dRate in enumerate(dRateArray):
        for i,tRate in enumerate(tRateArray):
        
            # U[0]: LOW RISK TESTING RATE
            # U[1]: HIGH RISK TESTING RATE
            # U[2]: LOW RISK DISTANCING RATE
            # U[3]: HIGH RISK DISTANCING RATE
            u[0] = tRate
            u[1] = tRate
            u[2] = dRate
            u[3] = dRate
        
            for k in range(len(Imm)): # IMMUNITY RATE
                
                S = (1-Imm[k])*np.array(N)
                NA = 1.*S
 
                # Cost for testing (low risk)
                costTmp = (u[0]>0)*a[0,0]+a[0,1]*u[0]*NA[0]*u[0]+a[0,2]*NA[0]*u[0]**2
                # Cost for testing (high risk)
                costTmp = costTmp+(u[1]>0)*a[1,0]+a[1,1]*u[1]*NA[1]*u[1]+a[1,2]*NA[1]*u[1]**2

                # TOTAL TESTING COST
                testCost[i,j,k] = costTmp

                
                # Cost for social distance (low risk)
                costTmp = b[0,0]*N[0]+b[0,1]*N[0]*u[2]+b[0,2]*N[0]*u[2]**2
                # Cost for social distance (high risk)
                costTmp = costTmp + b[1,0]*N[1]+b[1,1]*N[1]*u[3]+b[1,2]*N[1]*u[3]**2

                # TOTAL DISTANCE COST
                distCost[i,j,k] = costTmp

                # TOTAL COST
                costVal[i,j,k] = testCost[i,j,k]+distCost[i,j,k]


                # COLLECT THE RO INFO FROM THE VARIED PARAMETERS

                rhoValBeta[i,j,k,:] = getRhoVal('beta',betaArray,dictVar,u,S)
                rhoValTau[i,j,k,:] = getRhoVal('tau',tauArray,dictVar,u,S)
                rhoValWA[i,j,k,:] = getRhoVal('wA',wAArray,dictVar,u,S)
                
            #print(i,j,distCost[i,j,0],testCost[i,j,0])
        #aaaa=7



    # ### CREATE PLOTS 
    # ################################ 

    # # CREATE A SEPARATE FIGURE FOR EACH IMM  
        
    xmin = 0
    xmax = v1max
    ymin = 0
    ymax = u1max



    limits = [xmin,xmax,ymin,ymax]

    Rlevel = [1.5,1]

    for ii, imm in enumerate(Imm):
        plt.figure(ii, (9,6))
    
        pos1 =  plt.contourf(costVal[:,:,0],
                            extent = limits,levels=15)     
                            
        cbar1 = plt.colorbar(pos1) 
        cbar1.set_label('Cost (US Dollars/day)')
    
    
    
        CS0 = plt.contour(rhoValBeta[:,:,ii,1],levels=[Rlevel[ii]],colors = 'lightgrey',\
                          linewidths=2.5, extent = limits) 
        CS0.collections[0].set_label("$R_e={{{}}}$, baseline".format('{:.1f}'.format(Rlevel[ii])) )    
    
    
        #Beta R0
    
        CS1 = plt.contour(rhoValBeta[:,:,ii,0],levels=[Rlevel[ii]], colors = 'darkred',\
                          linewidths=2,linestyles = 'dashdot', extent = limits)  
        CS2 = plt.contour(rhoValBeta[:,:,ii,2],levels=[Rlevel[ii]], colors = 'darkred',\
                          linewidths=2,linestyles = 'dashed', extent = limits)    
    
        CSList = [CS1,CS2]
        
        for CS,rate in zip(CSList,[0.75,1.25]):
            CS.collections[0].set_label("$R_e={{{}}},~\\beta \\times{{{}}}$".format('{:.1f}'.format(Rlevel[ii]),'{:04.2f}'.format(rate)))
        
        
        #Tau R0
    
        CS1 = plt.contour(rhoValTau[:,:,ii,0],levels=[Rlevel[ii]], colors = 'darkorange',\
                          linewidths=2,linestyles = 'dashdot', extent = limits)  
        CS2 = plt.contour(rhoValTau[:,:,ii,2],levels=[Rlevel[ii]], colors = 'darkorange',\
                          linewidths=2,linestyles = 'dashed', extent = limits)    
    
        CSList = [CS1,CS2]
        for CS,rate in zip(CSList,[0.75,1.25]):
            CS.collections[0].set_label("$R_e={{{}}},~\\tau \\times {{{}}}$".format('{:.1f}'.format(Rlevel[ii]),'{:04.2f}'.format(rate)))
        
        
        #wA R0
    
        CS1 = plt.contour(rhoValWA[:,:,ii,0],levels=[Rlevel[ii]], colors = 'magenta',\
                          linewidths=2,linestyles = 'dashdot', extent = limits)  
        CS2 = plt.contour(rhoValWA[:,:,ii,2],levels=[Rlevel[ii]],colors = 'magenta',\
                          linewidths=2,linestyles = 'dashed', extent = limits)    
    
        CSList = [CS1,CS2]
        for CS,rate in zip(CSList,[0.75,1.25]):
            CS.collections[0].set_label("$R_e={{{}}},~\omega^A \\times {{{}}} $".format('{:.1f}'.format(Rlevel[ii]),'{:04.2f}'.format(rate)))    
    
    
    
        plt.title("$R_e$ Sensitivities @ {:.1f}% Immunity".format(imm*100),fontsize = 14)
        plt.xlabel('Social distance control level',fontsize = 14)
        plt.ylabel('Testing control level',fontsize = 14)
        plt.tick_params(axis='both', which='major', labelsize=14)
    
        plt.legend(bbox_to_anchor=(1.25,0.5), loc="center left", borderaxespad=0)
    
        plt.subplots_adjust(left = 0.1,right=0.79)


    plt.show()


        
#############################################
### THESE METHODS DEPEND ON EACH OTHER ######
    
# CREATE PLOTS FOR DIFFERENT DISTANCE/TEST COST

def getCostVal(dictVar,imm,u,param = None,rateArray = [1]):
    
    

    # INITIAL CONDITION
    
    N = dictVar['N']
    NA = dictVar['NA']
    
    
    
    distCost = np.zeros(len(rateArray))
    testCost = np.zeros(len(rateArray))
    
    
    
    S = (1-imm)*np.array(N)
    NA = 1.*S
    # Costs

    a_arr = np.array([[0,2.3,27],[0,2.3,27]]) # Testing costs
    b_arr = np.array([[0,0,40],[0,0,40]]) # Distancing costs
    
    a = 0
    b = 0
    
    for i, rate in enumerate(rateArray):
        
        a = np.copy(a_arr)
        b = np.copy(b_arr)
        
        
        if param == 'a':
            
            a[0,2] *= rate
            a[1,2] *= rate        
        elif param == 'b':
            
            b[0,2] *= rate
            b[1,2] *= rate
            
        
        # Cost for social distance (low risk)
        dCost = b[0,0]*N[0]+b[0,1]*N[0]*u[2]+b[0,2]*N[0]*u[2]**2
        # Cost for social distance (high risk)
        dCost = dCost + b[1,0]*N[1]+b[1,1]*N[1]*u[3]+b[1,2]*N[1]*u[3]**2
        
        # TOTAL DISTANCE COST
        distCost[i] = dCost
        
        
        # Cost for testing (low risk)
        tCost = (u[0]>0)*a[0,0]+a[0,1]*NA[0]*u[0]+a[0,2]*NA[0]*u[0]**2
        # Cost for testing (high risk)
        tCost = tCost+(u[1]>0)*a[1,0]+a[1,1]*NA[1]*u[1]+a[1,2]*NA[1]*u[1]**2
        
        # TOTAL TESTING COST
        testCost[i] = tCost
    
        
    return distCost,testCost
    

def distance_testing_multiplier_contour():
    
    dictVar = initialization()
    
    # INITIAL CONDITION
    
    N = dictVar['N']
    NA = dictVar['NA']
    xI = dictVar['xI']
    
    
    # SYSTEM PARAMETERS
    
    beta = dictVar['beta']
    Phi = dictVar['Phi']
    gamA = dictVar['gamA']
    gamY = dictVar['gamY']
    gamH = dictVar['gamH']
    eta = dictVar['eta']
    tau = dictVar['tau']
    sig = dictVar['sig']
    rhoA = dictVar['rhoA']
    rhoY = dictVar['rhoY']
    P = dictVar['P']
    wY = dictVar['wY']
    wA = dictVar['wA']
    IFR = dictVar['IFR']
    YFR = dictVar['YFR']
    YHR = dictVar['YHR']
    HFR = dictVar['HFR']
    wP = dictVar['wP']
    wPY = dictVar['wPY']
    wPA = dictVar['wPA']
    Pi = dictVar['Pi']
    mu = dictVar['mu']
    theta = dictVar['theta']
    nu = dictVar['nu']

        

    Imm = [0.0,0.666] # IMMUNITY RATE OF INDIVIDUALS

    
    tStep = 0.025
    dStep = 0.025

    # a = np.array([[0,2.3,27],[0,2.3,27]]) # Testing costs
    # b = np.array([[0,0,40],[0,0,40]]) # Distancing costs
    u1max = 0.66
    v1max = 0.8
    
    Rlevel = [1.5,1]

    # HIGH/LOW TEST RATE ARRAY
    tRateArray = np.arange(0,u1max,tStep)

    # HIGH/LOW DISTANCE RATE ARRAY
    dRateArray = np.arange(0,v1max,dStep)

    # SIZE OF THE HIGH/LOW TEST RATE ARRAY
    nTRA = len(tRateArray)

    # SIZE OF THE HIGH/LOW DISTANCE RATE ARRAY
    nDRA = len(dRateArray)


    rateArray = np.array([0.75,1.25])

    rhoVal = np.zeros((nTRA, nDRA,len(Imm))) # REPRODUCTION NUMBER
    costVal = np.zeros((nTRA,nDRA,len(Imm))) # COST
    distCost = np.zeros((nTRA,nDRA,len(Imm)))
    testCost = np.zeros((nTRA,nDRA,len(Imm)))
    
    a_distCost = np.zeros((nTRA,nDRA,len(Imm), len(rateArray)))
    a_testCost = np.zeros((nTRA,nDRA,len(Imm), len(rateArray)))
    
    b_distCost = np.zeros((nTRA,nDRA,len(Imm), len(rateArray)))
    b_testCost = np.zeros((nTRA,nDRA,len(Imm), len(rateArray)))
    
    
    #mulTest_costVal = np.zeros((nTRA,nDRA,len(Imm), len(rateArray)))
    #mulDist_costVal = np.zeros((nTRA,nDRA,len(Imm)), len(rateArray))

    u = np.zeros(4)    # [0,0,0,0] * CURRENT CONTROL RATE


    for j, dRate in enumerate(dRateArray):
        for i,tRate in enumerate(tRateArray):
        
            # U[0]: LOW RISK TESTING RATE
            # U[1]: HIGH RISK TESTING RATE
            # U[2]: LOW RISK DISTANCING RATE
            # U[3]: HIGH RISK DISTANCING RATE
            u[0] = tRate
            u[1] = tRate
            u[2] = dRate
            u[3] = dRate
        
            for k in range(len(Imm)): # IMMUNITY RATE
                
                    
                S = (1-Imm[k])*np.array(N)


                # GET TOTAL COST FOR DISTANCING AND TESTING
                totalDistCost, totalTestCost = getCostVal(dictVar, Imm[k], u)

                # TOTAL DISTANCE COST
                distCost[i,j,k]  = totalDistCost
                
                # TOTAL TESTING COST
                testCost[i,j,k] = totalTestCost
                
                # DISTANCE COST + TESTING COST
                costVal[i,j,k] = totalDistCost + totalTestCost
                
                
                a_distCost[i,j,k,:], a_testCost[i,j,k,:]  =  getCostVal(dictVar, Imm[k], u, param = 'a',rateArray= rateArray)
                b_distCost[i,j,k,:], b_testCost[i,j,k,:]  =  getCostVal(dictVar, Imm[k], u, param = 'b',rateArray= rateArray)
                    
                    
                
        
                # CALCULATE R0
            
      
                F00 = np.array(\
                      [np.array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])\
                             *(1-u[2])*beta*S[0]/N[0]*Phi[0,0],
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
                
                F = np.bmat([[F00,F01],[F10,F11]])
                V = np.bmat([[V00,V01],[V10,V11]])
                V_inv = np.linalg.inv(V)
                Prod = F*V_inv
                val,__ = speigs(Prod,k=1)
                rhoVal[i,j,k]=np.real(val[0])
                
    
    #### Create Plots
   

    for ii, imm in enumerate(Imm):
        
        plt.figure(ii, (9,6))
        
        xmin = np.min(distCost[:,:,ii])
        xmax = np.max(distCost[:,:,ii])
        ymin = np.min(testCost[:,:,ii])
        ymax = np.max(testCost[:,:,ii])
        
        limits = [xmin,xmax/2,ymin,ymax]
        
        # INITIAL CONDITIONS
        
        pos1 = plt.contourf(distCost[:,:,ii],
                             testCost[:,:,ii],
                             costVal[:,:,ii],
                             extent = limits,
                             levels = 15)
    
            
        
        cbar1 = plt.colorbar(pos1) 
        cbar1.set_label('Cost ($USD / day)')
        
        
        # BASELINE R0
        CS0 = plt.contour(distCost[:,:,ii],
                        testCost[:,:,ii],
                        rhoVal[:,:,ii],
                        levels = [Rlevel[ii]*.8,Rlevel[ii]],
                        colors = 'lightgrey',
                        linewidths=2)
    
        #        CS0.collections[0].set_label("$R_e = {}$, Baseline".format('{:.1f}'.format(Rlevel[ii])))  
        CS0.collections[0].set_label("Baseline")  

 
    
        ##################
        # MULTIPLIER FOR A
        ##################
    
        CS1 = plt.contour(a_distCost[:,:,ii,0],
                        a_testCost[:,:,ii,0],
                        rhoVal[:,:,ii],
                        levels = [Rlevel[ii]*.8,Rlevel[ii]],
                        colors = 'red',
                        linestyles = 'dashdot',
                        linewidths=2)
    
        #        CS1.collections[0].set_label("$R_e={},~a_{{j2}} \\times 0.75$".format('{:.1f}'.format(Rlevel[ii])))
        CS1.collections[0].set_label("$a_{{j2}} \\times 0.75$")
    
        
    
        CS2 = plt.contour(a_distCost[:,:,ii,1],
                        a_testCost[:,:,ii,1],
                        rhoVal[:,:,ii],
                        levels = [Rlevel[ii]*.8,Rlevel[ii]],
                        colors = 'red',
                        linestyles = 'dashed',
                        linewidths=2)
    
        #        CS2.collections[0].set_label("$R_e={},~a_{{j2}} \\times 1.25$".format('{:.1f}'.format(Rlevel[ii])))  
        CS2.collections[0].set_label("$a_{{j2}} \\times 1.25$")  
     
    
        ##################
        # MULITPLIER FOR B
        ##################
        
        CS1 = plt.contour(b_distCost[:,:,ii,0],
                    b_testCost[:,:,ii,0],
                    rhoVal[:,:,ii],
                    levels = [Rlevel[ii]*.8,Rlevel[ii]],
                    colors = 'cyan',
                    linestyles = 'dashdot',
                    linewidths=2)
    
        #        CS1.collections[0].set_label("$R_e={},~b_{{j2}} \\times 0.75$".format('{:.1f}'.format(Rlevel[ii])))  
        CS1.collections[0].set_label("$b_{{j2}} \\times 0.75$")  
     
    
        CS2 = plt.contour(b_distCost[:,:,ii,1],
                        b_testCost[:,:,ii,1],
                        rhoVal[:,:,ii],
                        levels = [Rlevel[ii]*.8,Rlevel[ii]],
                        colors = 'cyan',
                        linestyles = 'dashed',
                        linewidths=2)
    
        #        CS2.collections[0].set_label("$R_e={},~b_{{j2}} \\times 1.25$".format('{:.1f}'.format(Rlevel[ii])))  
        CS2.collections[0].set_label("$b_{{j2}} \\times 1.25$")  
     
    
        ##################
        # GENERAL
        ##################
        
    
        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))
        
        plt.title("Cost Sensitivities @ Immunity Level {:.1f}%".format(imm*100),fontsize = 14)
        plt.xlabel('Social distance cost ($USD / day)',fontsize = 14)
        plt.ylabel('Testing cost ($USD / day)',fontsize = 14)
        plt.tick_params(axis='both', which='major', labelsize=14)
    
        plt.legend(bbox_to_anchor=(1.25,0.5), loc="center left", borderaxespad=0)
    
        plt.subplots_adjust(left = 0.1,right=0.79)



    plt.show()






if __name__ == "__main__":

    #distance_testing_param_contour()

    distance_testing_multiplier_contour()








