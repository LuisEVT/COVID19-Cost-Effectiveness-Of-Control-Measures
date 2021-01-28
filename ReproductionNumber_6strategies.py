from  numpy import *
from scipy.sparse.linalg import eigs as speigs
from matplotlib.pyplot import * 



tau = 0.57 #from paper
IFR = array([0.6440/100,6.440/100])
YFR = array([IFR[0]/tau,IFR[1]/tau]) 
YHR = array([4.879/100,48.79/100])
HFR = array([YFR[0]/YHR[0],YFR[1]/YHR[1]])
P = 0.44 #from paper
wY =1. #from paper
wA =0.66 #from paper
rhoA = 1/2.3 ###rhoA=rhoY##1/rhoY~2.3#from paper
rhoY = 1/2.3 ###1/rhoY~2.3#from paper
gamA = 1/4 ####1/gamA~4#from paper
gamY = 1/4 ###gamA=gamY##1/gamA~4#from paper
eta = 0.1695 #from paper
wP= P/(1-P) /(tau*wY/rhoY + (1-tau)*wA/rhoA) \
    * ((1-tau)*wA/gamA \
    + tau*wY* array([YHR[0]/eta+(1-YHR[0])/gamY, \
                      YHR[1]/eta+(1-YHR[1])/gamY])) 
     
wPY = wP*wY # from paper
wPA = wP*wA #from paper
beta = 0.0640 #from paper
sig = 1/2.9 ###1/sig~2.9# from paper

Pi = gamY*array([YHR[0]/(eta+(gamY-eta)*YHR[0]),\
                 YHR[1]/(eta+(gamY-eta)*YHR[1])])# from paper

gamH = 1/10.7 #from paper
mu = 1/8.1 ###1/mu~8.1#from paper
mup = 0.0005 #2352 ventilators in Houston (https://www.click2houston.com/health/2020/04/10/texas-medical-center-data-shows-icu-ventilator-capacity-vs-usage-during-coronavirus-outbreak/)
nu = gamH*array([HFR[0]/(mu+(gamH-mu)*HFR[0]),\
             HFR[1]/(mu+(gamH-mu)*HFR[1])]) ##from paper
N = array([1340000, 423000])    
Phi = array([[10.56,2.77],[9.4,2.63]]) ###Contact matrix
umax = [.66,.66,.8,.8]
Imm = [0,0.666]

a = array([[0,2.3,27],[0,2.3,27]]) # Testing costs
b = array([[0,0,40],[0,0,40]]) # Distancing costs

iVal = range(100)
nRun = 6  

uArr =  zeros((nRun,100))               
rhoVal = zeros((len(iVal),nRun,len(Imm))) # Reproduction number
costVal = zeros((len(iVal),nRun,len(Imm))) # Cost

for i in iVal:                 
    iFrac = (iVal[i]+1)/(max(iVal)+1)
    for j in range(nRun):
        u=zeros(nRun)
        if j<4:
            u[j]=umax[j]*iFrac
            uArr[j,i]=umax[j]*iFrac
        elif j==4:
            u[0]=umax[0]*iFrac
            u[1]=umax[0]*iFrac
            uArr[j,i]=umax[0]*iFrac
        else:
            u[2]=umax[2]*iFrac
            u[3]=umax[2]*iFrac
            uArr[j,i]=umax[2]*iFrac
        
        for k in range(len(Imm)):
            S = (1-Imm[k])*array(N)
            NA = 1.*S
            # Costs
                        
            # Cost for testing (low risk)
            costTmp = (u[0]>0)*a[0,0]+a[0,1]*NA[0]*u[0]+a[0,2]*NA[0]*u[0]**2
            # Cost for testing (high risk)
            costTmp = costTmp+(u[1]>0)*a[1,0]+a[1,1]*NA[1]*u[1]+a[1,2]*NA[1]*u[1]**2
            # Cost for social distance (low risk)
            costTmp = costTmp + b[0,0]*N[0]+b[0,1]*N[0]*u[2]+b[0,2]*N[0]*u[2]**2
            # Cost for social distance (high risk)
            costTmp = costTmp + b[1,0]*N[1]+b[1,1]*N[1]*u[3]+b[1,2]*N[1]*u[3]**2
            costVal[i,j,k] = costTmp
    
        
            F00 = array(\
                  [array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])\
                         *(1-u[2])*beta*S[0]/N[0]*Phi[0,0],
                    [(1-tau)*sig,0,0,0,0],
            		[tau*sig,0,0,0,0],
                    [0,rhoA,0,0,0],
                    [0,0,rhoY,0,0]])
            
            F11 = array(\
                  [array([0,(1-u[1])*wPA[0],(1-u[1])*wPY[0],(1-u[1])*wA,wY])\
                        *(1-u[3])*beta*S[1]/N[1]*Phi[1,1],
                    [(1-tau)*sig,0,0,0,0],
                    [tau*sig,0,0,0,0],
                    [0,rhoA,0,0,0],
                    [0,0,rhoY,0,0]])
            
            F01 = array(\
                  [array([0,(1-u[1])*wPA[1],(1-u[1])**wPY[1],(1-u[1])*wA,wY])\
                        *(1-u[2])*beta*S[0]/N[1]*Phi[0,1],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])
            
            F10 = array(\
                  [array([0,(1-u[0])*wPA[0],(1-u[0])*wPY[0],(1-u[0])*wA,wY])\
                        *(1-u[3])*beta*S[1]/N[0]*Phi[1,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0]])
            
            V00 = array([[sig,0,0,0,0],
                         [0,rhoA,0,0,0],
                         [0,0,rhoY,0,0],
                         [0,0,0,gamA,0],
                         [0,0,0,0,(1-Pi[0])*gamY+Pi[0]*eta]])
            
            V11 = array([[sig,0,0,0,0],
                         [0,rhoA,0,0,0],
                         [0,0,rhoY,0,0],
                         [0,0,0,gamA,0],
                         [0,0,0,0,(1-Pi[1])*gamY+Pi[1]*eta]])
            
            V10 = zeros((5,5))
            V01 = copy(V10)
            
            F = bmat([[F00,F01],[F10,F11]])
            V = bmat([[V00,V01],[V10,V11]])
            V_inv = linalg.inv(V)
            Prod = F*V_inv
            val,__ = speigs(Prod,k=1)
            rhoVal[i,j,k]=real(val[0])
        

fig1,ax1 = subplots()
ax1.set_xlabel('Control level')                                
ax1.set_ylabel('Effective reproduction number')
ax1.set_prop_cycle(color = ['blue','green','red','orange','cyan','magenta'])

for jRun in range(nRun):
    ax1.plot(uArr[jRun,:],rhoVal[:,jRun,0])
for jRun in range(nRun):
    ax1.plot(uArr[jRun,:],rhoVal[:,jRun,1],'--')

#ax1.plot(uVal,rhoVal[:,:,2],':')                           
ax1.legend(["testing, low risk","testing, high risk",\
            "distancing, low risk","distancing, high risk",\
                "both testing","both distancing"],\
            bbox_to_anchor=(0, 0.37), loc='upper left')

fig2,ax2 = subplots()
ax2.set_xlabel('Control level')                                
ax2.set_ylabel('Control cost(US Dollars/Day)')
ax2.set_prop_cycle(color = ['blue','green','red','orange','cyan','magenta'])
for jRun in range(nRun):
    ax2.plot(uArr[jRun,:],costVal[:,jRun,0])
#for jRun in range(nRun):
#    ax2.plot(uArr[jRun,:],costVal[:,jRun,1],'--')
#ax2.plot(uVal,costVal[:,:,2],':')                           
ax2.legend(["testing, low risk","testing, high risk",\
            "distancing, low risk","distancing, high risk",\
            "both testing","both distancing"],\
            loc='upper left')
#ax2.set_yscale('log')

fig3,ax3 = subplots()
ax3.set_ylabel('Effective reproduction number')                                
ax3.set_xlabel('Control cost(US Dollars/Day)')
ax3.set_prop_cycle(color = ['blue','green','red','orange','cyan','magenta'])
ax3.plot((costVal[:,:,0]),rhoVal[:,:,0])
ax3.plot((costVal[:,:,1]),rhoVal[:,:,1],'--')
#ax3.plot((costVal[:,:,2]),rhoVal[:,:,2],':')                           
ax3.legend(["testing, low risk","testing, high risk",\
            "distancing, low risk","distancing, high risk",\
                "both testing","both distancing"],\
            loc='upper right')
#ax3.set_xscale('log')

show()
#############
