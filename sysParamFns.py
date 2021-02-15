from  numpy import *
# This is the function for the solution x(t) (18 components)

def Afn(t,x,params):
    wY,wA,wPY,wPA,beta,sig,tau,rhoA,rhoY,gamA,gamY,gamH,Pi,eta,nu,mu,theta,rr,Phi,u= params  
    out = zeros(len(x))
    N0inv = 1/(sum(x[0:9])+1); # Add 1 to avoid division by 0
    N1inv = 1/(sum(x[9:18])+1); # Add 1 to avoid division by 0
    infect0 = N0inv*(x[5]*wY + \
              (1-u[0])*x[4]*wA+ (1-u[0])*x[3]*wPY[0]+(1-u[0])*x[2]*wPA[0]);
    infect1 = N1inv*(x[14]*wY + \
              (1-u[1])*x[13]*wA+ (1-u[1])*x[12]*wPY[1]+(1-u[1])*x[11]*wPA[1]);
    # Increased deaths due to respirators running out
    respParam = rr*(1 -  theta/rr/max(nu[0]*x[6]+nu[1]*x[15],theta/rr))    
    out[0] = -infect0*(1-u[2])*beta*x[0]*Phi[0,0]###
    out[0] = out[0] - infect1*(1-u[3])*beta*x[0]*Phi[0,1]###
    out[1] = -out[0] - sig*x[1]
    out[2] = (1-tau)*sig*x[1] - rhoA*x[2]
    out[3] = tau*sig*x[1] - rhoY*x[3]
    out[4] = rhoA*x[2] - gamA*x[4]
    out[5] = rhoY*x[3] - (1 - Pi[0])*gamY*x[5] - Pi[0]*eta*x[5]
    out[6] = Pi[0]*eta*x[5] - (1-nu[0])*gamH*x[6] - mu*nu[0]*x[6]
    out[7] = gamA*x[4] + (1-Pi[0])*gamY*x[5] +(1-nu[0]*(1+respParam))*gamH*x[6]
    out[8] = mu*nu[0]*(1 + respParam)*x[6]
    out[9] = -infect1*(1-u[3])*beta*x[9]*Phi[1,1]###
    out[9] = out[9] - infect0*(1-u[2])*beta*x[9]*Phi[1,0]###
    out[10] = -out[9]-sig*x[10]
    out[11] = (1-tau)*sig*x[10] - rhoA*x[11] 
    out[12] = tau*sig*x[10] - rhoY*x[12]
    out[13] = rhoA*x[11] - gamA*x[13]
    out[14] = rhoY*x[12] - (1 - Pi[1])*gamY*x[14] - Pi[1]*eta*x[14]
    out[15] = Pi[1]*eta*x[14] - (1-nu[1])*gamH*x[15] - mu*nu[1]*x[15]
    out[16] = gamA*x[13] + (1-Pi[1])*gamY*x[14] +(1-nu[1]*(1+respParam))*gamH*x[15]
    out[17] = mu*nu[1]*x[15]*(1+respParam)

    return out
    