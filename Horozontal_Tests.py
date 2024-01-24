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


Tsn = []



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
    s1 = 0.22 * kappa_m * (T_1 - T_s) * (Ra_m**(2.0 / 7.0)) * (Pr**(-1.0 / 7.0))
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
    s1 = 0.22 * kappa_m * (T_c - T_2) * (Ra_c**(2.0 / 7.0)) * (Pr**(-1.0 / 7.0))
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



def Ra_H(T_s, T_sn):

    s1 = alpha_m * rho * g * (T_s - T_sn) * ((pi * R_p)**3.0)
    s2 = k_c * eta_m
    s3 = s1 / s2

    return s3



def Q_H(T_s, T_sn):

    Ra_h = Ra_H(T_s, T_sn)
    s1 = 0.22 * kappa_m * (T_s - T_sn)
    s2 = pi * R_p
    s3 = (s1 / s2) * (Ra_h**(2.0 / 7.0)) * (Pr**(-1.0 / 7.0))

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
T_e = 0.0  # Equilibrium temperature
T_s = 3800.00  # Surface temperature dayside
T_sn = 1600.00  # Surface temperature nightside
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


    

for t in range(1, 100000, 1):
    
    


    V_m = (4.0 / 3.0) * ((pi * R_p**3.0) - (pi * R_c**3.0))

    Pr = (eta_m / rho) / k_c
    i += 1
    n = 100
    
    Q_h = Q_H(T_s, T_sn)
    dT_dt = ((Q_h * (pi * R_p**2.0)) / (rho * V_m * C)) * (3600 * 24 * 365 * n)

    T_s -= dT_dt
    T_sn += dT_dt

    if t % 2 == 0:

        time.append(t * n)
        Ts.append(T_s)
        Tsn.append(T_sn)


    
    

print(len(time), len(Ts), len(Tsn))

import matplotlib
matplotlib.rcParams.update({'font.size': 25})
matplotlib.rcParams['axes.linewidth'] = 2.2
plt.minorticks_on()
plt.tick_params(axis='x',direction='in', top=True, length=14.4, width=2.0)
plt.tick_params(axis='x', which='minor',top=True, direction='in', length=6.4, width=2.0)
plt.tick_params(axis='y',direction='in', right=True, length=14.4, width=2.0)
plt.tick_params(axis='y', which='minor', right=True, direction='in', length=6.4, width=2.0)


plt.loglog(time, Ts, color='red', label='$T_{s}$ dayside', linewidth=3.0)
plt.loglog(time, Tsn, color='orange', label='$T_{s}$ nightside', linewidth=3.0)
plt.xlabel('Time (yrs)')
plt.ylabel('Temperature (K)')
#plt.title('Thermal evolution without heat exchange')
plt.title('Thermal evolution with heat exchange')
plt.legend()
plt.show()







