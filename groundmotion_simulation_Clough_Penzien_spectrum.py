import math
import numpy as np

def groundmotion_simulation( S_o = 1, f_g = 3, d_g = 0.6, f_f = 0.3, d_f = 0.6, dt = 0.005, T = 20 ):

    npts = int( T / dt  ) + 1
    time_vector = np.linspace( 0,  T,  npts )
    fs = 1/dt
    nfft = 2 ** math.ceil( math.log2( npts ) )
    f_all = np.linspace( 0, fs / 2, int( nfft / 2+1 ) )

## stationary ground motion

# ASD spectrum estimation
    H_Kanai_Tajimi = np.divide((1 +  np.power(2*d_g*f_all/f_g, 2)), (np.power(1 - np.power(f_all/f_g, 2), 2) + np.power(2*d_g*f_all/f_g, 2)))
    H_Clough_Penziene = np.divide(np.power(f_all/f_f, 4), (np.power(1 - np.power(f_all/f_f, 2), 2) + np.power(2*d_f*f_all/f_f, 2)))
    S_gg = S_o*np.multiply(H_Kanai_Tajimi, H_Clough_Penziene)

# Random phase
    phase = 2*math.pi*np.random.rand( int( nfft / 2+1 ))

# Complex Fourier coefficients
    X_gg = np.multiply( np.power(T*S_gg, 0.5), np.exp(1j*phase))    
    X_gg = np.concatenate((X_gg, np.flipud(np.conjugate(X_gg[1: -1]))))

# ground acceleration throguh inverse Fourier transform
    grnd_acln = np.fft.ifft(X_gg, n = npts)/dt

## Envelop function E
    Envlpe = 0.906*np.multiply(time_vector, np.exp(-time_vector/3))

## Envelop multiplied with stationary process to get ground motion
    grnd_acln = np.multiply(grnd_acln, Envlpe)

    return time_vector, grnd_acln
