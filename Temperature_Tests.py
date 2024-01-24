import pylab
from math import *
from scipy.constants import G
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np


# Arrays

time = []; time_a = []
step_size = []; step_size_a = []

T1 = []; T1_a = []
T2 = []; T2_a = []
T3 = []; T3_a = []
T4 = []; T4_a = []
T5 = []; T5_a = []
T6 = []; T6_a = []
Ts = []; Ts_a = []
Tc = []; Tc_a = []
t_d = []
f_d = []


Qm = []; Qm_a = []
Qms = []; Qms_a = []
Qs = []; Qs_a = []
Qc = []; Qc_a = []
Qr = []; Qr_a = []


R2 = []; R2_a = []
R3 = []; R3_a = []
R4 = []; R4_a = []
R5 = []; R5_a = []
R6 = []; R6_a = []
Ri1 = []; Ri1_a = []
Ri2 = []; Ri2_a = []
Cl = []
Ml = []

surface_e = []; surface_e_a = []
insolation_e = []; insolation_e_a = []




# Constants

C_p = 850.0  
M_c = 1.97 * (10**24)
delta = 1.400
delta_rho = 1.260  #default = 0.3
L_H = 750000
kappa = 200.0
kappa_m = 13.20
kappa_m2 = 8.0
rho = 4600.00
alpha_c = 1.90e-05
alpha_m = 3.0e-06
k_c = 7.5 * (10**(-7.0))
Ra_crit = 450.0
C = 1250.00
sigma = ((5.67 * (10**(-8))))
D_ms = 8000.00 * (10**3.0)






def Liq(p):
    s1 = (p * 0.1169) + 1.0
    s2 = s1**(0.32726)
    return 2000.0 * s2


def Sol(p):
    s1 = (p * 0.0971) + 1.0
    s2 = s1**(0.351755)
    return 1674.0 * s2





def Q_rad(T_s, T_e):
    
    F = sigma * ((T_s**4.0) - (T_e**4.0))
    
    return F 



def dTs(T_s, T_e, r_a, r_b, Q_H):

    A_m = 4.0 * pi * (r_a**2.0)
    V_m = (4.0 / 3.0) * ((pi * r_a**3.0) - (pi * r_b**3.0))
    Q_r = Q_rad(T_s, T_e)
    Q_m = Q_MO(T_1, T_2, T_s, T_c, r_a, r_b)

    s1 = ((Q_m * A_m) - (Q_r * A_m)) + (Q_H * A_m) 
    s2 = V_m * C * rho
    s3 = s1 / s2

    return s3



def Ra_MO(T_1, T_2, T_s, T_c, r_a, r_b):

    s1 = (rho * g * alpha_m * (T_c - T_s - (T_2 - T_1)) * ((r_a - r_b)**3.0))
    s2 = (k_c * eta_m)
    s3 = s1 / s2

    return s3


def Q_MO(T_1, T_2, T_s, T_c, r_a, r_b):

    Ra_m = Ra_MO(T_1, T_2, T_s, T_c, r_a, r_b)
    s1 = 0.22 * kappa_m * 1.0 * (T_1 - T_s) * (Ra_m**(2.0 / 7.0)) * (Pr**(-1.0 / 7.0))
    s2 = (r_a - r_b)
    s3 = s1 / s2
    
    return s3


def dT1(T_1, T_2, T_s, T_c, r_a, r_b):

    A_m = 4.0 * pi * (r_a**2.0)
    A_c = 4.0 * pi * (R_c**2.0)
    V_m = (4.0 / 3.0) * ((pi * r_a**3.0) - (pi * r_b**3.0))
    Q_m = Q_MO(T_1, T_2, T_s, T_c, r_a, r_b)
    Q_c = Q_C(T_1, T_2, T_s, T_c, r_a, r_b)

    s1 = (Q_c * A_c) - (Q_m * A_m) 
    s2 = V_m * C * rho
    s3 = s1 / s2

    return s3


def Ra_C(T_1, T_2, T_s, T_c, r_a, r_b):

    s1 = (rho * g * alpha_m * (T_c - T_s - (T_2 - T_1)) * ((r_a - r_b)**3.0))
    s2 = (k_c * eta_m)
    s3 = s1 / s2

    return s3

    
def Q_C(T_1, T_2, T_s, T_c, r_a, r_b):

    Ra_c = Ra_C(T_1, T_2, T_s, T_c, r_a, r_b)
    s1 = 0.22 * kappa_m * 1.0 * (T_c - T_2) * (Ra_c**(2.0 / 7.0)) * (Pr**(-1.0 / 7.0))
    s2 = (r_a - r_b)
    s3 = s1 / s2

    return s3


def dTc(T_1, T_2, T_s, T_c, r_a, r_b):

    V_c = (4.0 / 3.0) * pi * R_c**3.0
    A_c = 4.0 * pi * (R_c**2.0)
    Q_c = Q_C(T_1, T_2, T_s, T_c, r_a, r_b)

    s1 = -1.0 * (Q_c * A_c)
    s2 = V_c * C * rho
    s3 = s1 / s2

    return s3





def R_2(T_2):

    P_2 = ((((T_2 / 2000.00)**(1.0 / 0.32726)) - 1.0) / 0.1169) # Pressure at R2 with T2
    P_2 = P_2 * 10**9
    r2 = R_p - (P_2 / (rho * g))
    
    return r2


def R_3(T_3):

    P_3 = ((((T_3 / 2000.00)**(1.0 / 0.32726)) - 1.0) / 0.1169) # Pressure at R2 with T2
    P_3 = P_3 * 10**9
    r3 = R_p - (P_3 / (rho * g))
    
    return r3





# Parameters of equations

#D_c = (((3.0 * C_p) / (2.0 * pi * alpha_c * rho_c * G))**(0.5))  # Temp. scale height

g = 10.4  # Gravity field in mantle

R_p = 6371.80 * (10**3.0)  # Radius of planet
R_c = 3480.00 * (10**3.0)  # Radius of core
R_c_l = 3482.00 * (10**3.0)  # Radius of core buffer

P_cmb = rho * g * (R_p - R_c_l)
P_cmb = P_cmb / (10**9)  # Pressure at CMB in GPa

T_c = 6400.00  # Temp. at CMB
T_e = 3000.0  # Equilibrium temperature
T_s = 3800.00  # Surface temperature
T_1 = 4000.00
T_2 = T_1 * np.exp((1.0*(R_p**1.0 - R_c_l**1.0)) / D_ms**1.0)
T_3 = T_2
T_4 = T_3
T_5 = T_4
T_6 = T_5
t_d.append((T_s + 1.0))
f_d.append(0.0)
eta_m = 0.1  # Viscosity of liquid in Pa.s
eta_ms = 10e13 # Viscosity of mush in Pa.s
eta_s = 10e17 # Viscosity of solid in Pa.s
D_m = 3600.00 * (10**3.0)
i = 0.0

Q_H = 1000000

    

for t in range(1, 8000000, 1):
    
    

    # Get radii

    r2 = R_2(T_2)


    # Get dT/dt for each layer

    Pr = (eta_m / rho) / k_c
    i += 1
    
    Q_r = Q_rad(T_s, T_e)
    Q_m = Q_MO(T_1, T_2, T_s, T_c, R_p, R_c)
    Q_c = Q_C(T_1, T_2, T_s, T_c, R_p, R_c)

    dT_s = dTs(T_s, T_e, R_p, R_c, Q_H)
    dT_1 = dT1(T_1, T_2, T_s, T_c, R_p, R_c)
    dT_c = dTc(T_1, T_2, T_s, T_c, R_p, R_c)


    # For the adaptive time step
    
        
    
    t_d.append(T_s)
    f_d.append(Q_r)
    i = int(i)
    flux_diff = abs(f_d[i-1] - f_d[i]) 
    temp_diff = t_d[i-1] - t_d[i]
    n = (temp_diff / flux_diff)
    step_size.append(n)

    #n = 100

    # For varying the temperature profile

    T_s += (dT_s * (3600 * 24 * 365 * n))
    T_1 += (dT_1 * (3600 * 24 * 365 * n))
    T_2 = T_1 * np.exp((R_p**1.0 - R_c**1.0) / D_ms**1.0)
    T_c += (dT_c * (3600 * 24 * 365 * n))
    

    time.append(t * n)
    T1.append(T_1)
    T2.append(T_2)
    Tc.append(T_c)
    Ts.append(T_s)
    
    Qm.append(Q_m)
    Qc.append(Q_c)
    Qr.append(Q_r)

    R2.append(r2)

    # Energy budget

    surf_en = 4.0 * pi * (R_p**2.0) * sigma * (T_s**4.0)
    inso_en = 4.0 * pi * (R_p**2.0) * sigma * (T_e**4.0)
    surface_e.append(surf_en)
    insolation_e.append(inso_en)





for i in range(0, len(time), 20):

    time_a.append(time[i])
    T1_a.append(T1[i])
    T2_a.append(T2[i])
    Tc_a.append(Tc[i])
    Ts_a.append(Ts[i])

    Qm_a.append(Qm[i])
    Qc_a.append(Qc[i])
    Qr_a.append(Qr[i])

    R2_a.append(R2[i])
    Cl.append(R_c_l)
    Ml.append(R_p)

    step_size_a.append(step_size[i])
    surface_e_a.append(surface_e[i])
    insolation_e_a.append(insolation_e[i])
    



import matplotlib
matplotlib.rcParams.update({'font.size': 25})
matplotlib.rcParams['axes.linewidth'] = 2.2
plt.minorticks_on()
plt.tick_params(axis='x',direction='in', top=True, length=14.4, width=2.0)
plt.tick_params(axis='x', which='minor',top=True, direction='in', length=6.4, width=2.0)
plt.tick_params(axis='y',direction='in', right=True, length=14.4, width=2.0)
plt.tick_params(axis='y', which='minor', right=True, direction='in', length=6.4, width=2.0)


plt.loglog(time_a, Ts_a, color='red', label='MO (T_1 = T_s)', linewidth=3.0)
plt.loglog(time_a, T1_a, color='orange', label='MO', linewidth=3.0)
plt.loglog(time_a, T2_a, color='orange', linestyle='--', linewidth=3.0)
plt.loglog(time_a, Tc_a, color='gold', label='CMB', linewidth=3.0)
plt.xlabel('Time (yrs)')
plt.ylabel('Temperature (K)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()


plt.loglog(time_a, Qr_a, color='orange', label='Qr', linewidth=3.0)
plt.loglog(time_a, Qm_a, color='orange', label='Qm', linestyle='--', linewidth=3.0)
plt.loglog(time_a, Qc_a, color='yellow', linestyle='--', label='CMB', linewidth=3.0)
plt.xlabel('Time (yrs)')
plt.ylabel('Power (W)')
plt.legend()
plt.show()


plt.loglog(time_a, surface_e_a, color='red', label='Total radiated energy', linewidth=3.0)
plt.loglog(time_a, insolation_e_a, color='green', linestyle='--', label='Total received energy', linewidth=3.0)
plt.xlabel('Time (yrs)')
plt.ylabel('Power (W)')
plt.legend()
plt.show()


plt.loglog(time_a, step_size_a)
plt.show()



plt.loglog(time_a, R2_a, color='black', label='R2')
#plt.loglog(time, Ri, color='black', label='R_inter')
#plt.loglog(time, R3, color='black', label='R3')
#plt.loglog(time, R4, color='black', label='R4')
#plt.loglog(time_a, Ri2_a, color='black', label='R_inter_2')
#plt.loglog(time, R5, color='black', label='R5')
plt.fill_between(time_a, Ml, color='orange')
plt.fill_between(time_a, R2_a, color='brown')
#plt.fill_between(time, Ri, alpha=0.77, color='brown')
#plt.fill_between(time, R3, color='brown')
#plt.fill_between(time, R4, facecolor='grey')
#plt.fill_between(time, Ri2, color='grey')  # Here
#plt.fill_between(time, R5, color='grey')
plt.axhline(y=6971000, color='black', label='', linestyle='--', linewidth=3.0)
plt.axhline(y=6371000, color='black', label='', linestyle='--', linewidth=3.0)
plt.axhline(y=3482000, color='black', label='', linestyle='--', linewidth=3.0)
plt.axhline(y=2990000, color='black', label='', linestyle='--', linewidth=3.0)
plt.fill_between(time_a, Cl, color='yellow')

#plt.text(32506, 3275000, 'CORE')
#plt.text(178985000, 4208070, 'SOLID')
#plt.text(8926, 4752890, 'MUSH')
#plt.text(186, 5572000, 'LIQUID')
#plt.text(32506, 6741000, 'SURFACE')
plt.xlabel('Time (yrs)')
plt.ylabel('Radius (m)')
#plt.legend()
#plt.title('Dayside: 1.0$R_{\oplus}$')
plt.show()



