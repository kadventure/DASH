"""DASH implementation"""
"""Kibum Nam, Alexander Jesacher
Inst. of Biomed. Physics, Innsbruck University, Austria

this script implements the DASH/c-DASH algorithms for aberration correction in 2-photon fluorescence microscopy.

"""


#%% loading packages & defining functions

import numpy as np
from numpy import pi
from matplotlib.pyplot import * 
from numpy.fft import fftshift, ifftshift, fft2, ifft2, fftfreq
#from mkl_fft import fft2, ifft2 #faster if you have an Intel CPU
from scipy.ndimage import gaussian_filter


def normalize(C):
    """normalizing complex array C"""
    if C.ndim == 2:
        C = C[None, :, :]
    
    N = np.sum(np.abs(C)**2, axis=(1,2))
    C_norm = C/np.sqrt(N[:, None, None])*np.sqrt(N_modes)  
    return C_norm, N 

def make_sample(type):
    """creating a fluorescence sample"""
    if type == "bead":
        sample = np.ones((N, N))*1e-6 
        sample[0,0] = 1 
    elif type == "plane":
        sample = np.ones((N,N))
    return sample

def measure(scat, sample, E, padding = True):
    """testing the correctin pattern C, i.e. measuring the 2-photon signal when it is applied to the SLM"""
       
    if E.ndim == 2:
        E = E[None, :, :]
    
    #optional padding to increase the spatial resolution in the focal plane
    if padding:
        tmp = fftshift(np.pad(ifftshift(scat*E, axes = (1,2)), [[0,0],[N,N],[N,N]]), axes = (1,2))
        sample = fftshift(np.pad(ifftshift(sample), N))
    else:    
        tmp = scat * E
    
    PSF = np.abs((fft2((tmp))) / N / N)**2  #intensity PSF in the focal plane
    TPEF = eta * np.sum(sample * PSF**2, axis = (1,2)) ##2-photon signal
    
    return TPEF, PSF

def create_scat(N, P2V, sigma) :
    """creating a random phase scatterer, defined by peak-to-valley P2V and sigma"""
    scat = np.random.random((N,N))
    scat = gaussian_filter(scat, sigma)
    scat -= np.min(scat)
    scat /= np.max(scat)
    
    scat = ifftshift(np.exp(1j* P2V * scat))
    
    
    figure()
    imshow(ifftshift(np.angle(scat)), cmap = "hsv", vmin=-pi, vmax=pi); 
    colorbar(); 
    title("scatterer")
    show()

    return scat

def create_modes(N):
    """creating the plane wave testmodes M"""
    # Create a grid of indices
    row, col = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # Create the delta function matrix
    delta = np.zeros((N, N, N, N), dtype=complex)
    delta[row, col, row, col] = 1.0
    
    # Compute the FFT of the delta function matrix
    M = fftshift(fft2(ifftshift(delta, axes=(2, 3)), axes=(2, 3)), axes=(2, 3))
    
    # Compute the frequency
    freq = np.sqrt((row - N // 2) ** 2 + (col - N // 2) ** 2).flatten()
    
    # Sort the modes by frequency
    idx = np.argsort(freq)
    M = M.reshape(N * N, N, N)[idx]
    
    return M

    
#%% create scatterer and modes

N = 25 # SLM & scattering media side length in pixels
N_modes = N**2  #no. of plane wave modes 
padding = True  #padding in Fourier plane to increase resolution in target plane

P2V = 1*pi  #peak to valley phase of scatterer
sigma_scat = 1.0    #gaussian blur kernel sigma of scatterer
scat = create_scat(N, P2V, sigma_scat) #calculate scatterer
M = create_modes(N) #calculating stack of plane-wave modes


#%% DASH user parameters

sample_type = "plane" # "bead" or "plane
method = "c-DASH" # "DASH" or "c-DASH"

N_i = 3  #number of iterations
N_p = 3 # number of phase steps
eta = 1e3 #2-photon efficieny
f = 0.2 #energy fraction of the testmodes
r0 = 1 #initial weight of the first mode
# r0 = 0.476 # (Optional) Setting Reduction Factor(r0) for the First mode 

sample = make_sample(sample_type) #define fluorescent sample: 2D plane or "bead"
w = np.zeros((N_i, N_modes), dtype=complex) #init. mode weights
theta = np.arange(N_p) * 2 * pi / N_p #define phase-stepping angles

# (Optional) Setting Abortion Criterion variables (Optional)
SlopeDetection = True
# SlopeDetection = False
N_test_modes = 24;  # number of examined signals
slope_percentage = 0.1  # abortion criterion
latest_sig = np.zeros(N_test_modes)  # abortion criterion
End_iter_detector = np.zeros([N_i])  # record when the abortion is activated
break_idx = -1 # counting idx for the abortion

m0 = 1 #start with measuring the second mode in the first DASH iteration (the weight of the first mode is directly calculated from an initial measurement)

if padding:
    N_pad = 3*N
else: 
    N_pad = N

w0 = r0*(measure(scat, sample, np.ones((N,N)), padding)[0]).item()**(1/4) #initial weight for 1st mode
w[0,0] = w0

C_i = np.zeros((N_i, N, N), dtype=complex) #init. stack containing the correction patterns after each iteration
I2ph_stack = np.zeros(N_i * (N**2) - m0) #init. stack of all TPEF signals
PSF_stack = np.zeros((N_i * (N**2) - m0, 1*N_pad, 1*N_pad)) #init. stack of all internsity PSFs
C = w0 * np.ones((N, N), dtype=complex)   #choice of initial correction pattern

#----- DASH loop-------------

mm = 0 #init. count index

for i in range(N_i):
    print("iteration = ", i)
      
    for m in range(m0,N_modes):
        
        m0 = 0 #from the 2nd iteration on, include a0 in the testing
            
        #A) phase stepping and measuring the signal
        if method == "DASH":
            E_SLM = np.exp(1j * np.angle(np.sqrt(f) * M[m][None, :,:] * np.exp(1j * theta[:,None, None]) + np.sqrt(1 - f) * normalize(C)[0]))
        elif method == "c-DASH":
            E_SLM = np.sqrt(f) * M[m][None, :, :] * np.exp(1j * theta[:, None, None]) + np.sqrt(1 - f) * normalize(C)[0] 
        
        I2ph = measure(scat, sample, E_SLM, padding)[0] #measuring TPEF signal for all phase steps -> vector I2ph
        S_avg = np.mean(np.sqrt(I2ph)) #average sqrt of 2-photon signal
        a  = np.sum(np.sqrt(I2ph) * np.exp(1j * theta)) / len(theta)   
          
        #B) calculating the mode weight
        w_m = np.sqrt(1/2/f * (S_avg - np.sqrt(S_avg**2 - 4*np.abs(a)**2))).item() #calculating optimal amplitude of mode
        w[i,m] = w_m * np.exp(1j*np.angle(a))

        #C) updating correction pattern
        if i > 0 and method == "c-DASH":
            C += (w[i, m] - w[i-1,m]) * M[m] #for c-DASH subtracting old mode gives better results
        else:
            C += w[i, m] * M[m]
        
        #D) testing the correction pattern 
        if method == "DASH":
            TPEF, PSF = measure(scat, sample, np.exp(1j*np.angle(C)), padding) #testing the correction pattern C
            
        elif method == "c-DASH":
            TPEF, PSF = measure(scat, sample, normalize(C)[0], padding) #testing the correction pattern C
        
        I2ph_stack[mm] = TPEF[0]
        PSF_stack[mm, :, :] = fftshift(PSF)

        # (Optional) Current Iteration Abortion Criterion 
        if SlopeDetection == True and m >= N_test_modes:
            latest_sig = I2ph_stack[(mm - N_test_modes+1):(mm+1)]  # lastest series of signals
            slope = (latest_sig[-1] - latest_sig[0]) / N_test_modes  # measure the slope
            if m == N_test_modes:  # measure the first slope
                slope0 = slope
            elif m > N_test_modes and slope < slope_percentage * slope0:  # compare current slope with the first slope
                break_idx += 1
                End_iter_detector[break_idx] = mm
                break
        mm += 1
    C_i[i] = C

#------showing results------

figure()
subplot(121)
title("Focal int. before corr.")
imshow(PSF_stack[0])
axis("off")
colorbar()
subplot(122)
imshow(PSF_stack[mm])
colorbar()
axis("off")
title("Focal int. after corr.")
show()

figure(figsize=(12, 5))
subplot(131)
imshow(ifftshift(np.angle(scat * C)), cmap='hsv', vmin = -pi, vmax = pi)
colorbar()
title('Scatt. phase + angle(C)')
axis('image')

if method == "c-DASH":
    subplot(132)
    imshow(ifftshift(np.abs(C)), cmap='gray', vmin = 0)
    colorbar()
    title('abs(C)')
    axis('image')

I2ph_max = measure(1, sample, np.ones((N,N)), padding)[0].item()  #eval. max. signal for the aberration-free case

figure()
if SlopeDetection == True:  # If Abortion Criterion is used
    plot(I2ph_stack[:mm])
    ylim(0, I2ph_max*1.1)
    for i in range(break_idx+1):    # Draw end of iterations
        axvline(x=End_iter_detector[i], color='r', linestyle='--', linewidth=1)
    axhline(y = I2ph_max, color='g', linestyle='--')
    title(f"Abort Criterion is applied. \n Number of measurements when each iteration ends: \n {End_iter_detector}")
else:
    plot(I2ph_stack)
    ylim(0, I2ph_max*1.1)
    xlim(0, N_i * (N**2) - 1)
    xlabel('Mode measurement no.')
    ylabel('2 photon signal / photons')
    axvline(x=N_modes-1, color='r', linestyle='--')
    axvline(x=2*N_modes-1, color='r', linestyle='--')
    axhline(y = I2ph_max, color='g', linestyle='--')
    title(method + ", " + str(N_modes) + " modes , sample = " + sample_type)
show()

