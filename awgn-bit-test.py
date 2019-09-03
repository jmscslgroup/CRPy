"""
awgn function provided from https://github.com/veeresht/CommPy/blob/master/commpy/channels.py, the rest was made from scratch

Author: Brandon Dominique
Email: dominiquebdmnq@aol.com
Date:7/17/19
*** all of this code was moved to mpsk.py, updated versions of all of this is there.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
def digital2analog(bit_array):
    start = 0
    stop = 0.01
    number_of_steps = 100
    
    time_for_individual_signal = np.linspace(start,stop,num = number_of_steps)
    t = time_for_individual_signal
    
    time_for_entire_modulated_signal = np.linspace(start,stop*2*len(bit_array),num = number_of_steps*2*len(bit_array))
    big_t = time_for_entire_modulated_signal

    time_for_part_of_signal = np.linspace(start,stop*len(bit_array),num = number_of_steps*len(bit_array))
    half_t = time_for_part_of_signal

    
    signal = []
    inphase = np.reshape(np.real(mod),(len(mod),1))*np.reshape(np.cos(2*np.pi*10000*t),(1,number_of_steps))
    quadrature = np.reshape(np.imag(mod),(len(mod),1))*np.reshape(np.sin(2*np.pi*10000*t),(1,number_of_steps))
    signal = np.concatenate((inphase,quadrature),axis = 0)
    
    plt.subplot(3,1,1)
    plt.title('Modulated Bandpass Signal (Real Part)')
    plt.plot(half_t,inphase.flatten())
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.subplot(3,1,2)
    plt.title('Modulated Bandpass Signal (Imaginary Part)')
    plt.plot(half_t,quadrature.flatten())
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.subplot(3,1,3)
    plt.title('Entire Modulated Bandpass Signal')
    plt.plot(big_t,signal.flatten())
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.show()
    return signal
                
        
def awgn(input_signal, snr_dB, rate=1.0):
    """
    Addditive White Gaussian Noise (AWGN) Channel.
    Parameters
    ----------
    input_signal : 1D ndarray of floats
        Input signal to the channel.
    snr_dB : float
        Output SNR required in dB.
    rate : float
        Rate of the a FEC code used if any, otherwise 1.
    Returns
    -------
    output_signal : 1D ndarray of floats
        Output signal from the channel with the specified SNR.
    """
    if len(np.shape(input_signal)) == 2:
        x,y = np.shape(input_signal)
        time_for_entire_modulated_signal = np.linspace(start,stop*x,num = number_of_steps*x)
        big_t = time_for_entire_modulated_signal

    avg_energy = np.sum(np.abs(input_signal) * np.abs(input_signal)) / len(input_signal)
    snr_linear = 10 ** (snr_dB / 10.0)
    noise_variance = avg_energy / (2 * rate * snr_linear)

    if isinstance(input_signal[0], complex):
        noise = (math.sqrt(noise_variance) * np.random.randn(len(input_signal))) + (math.sqrt(noise_variance) * np.random.randn(len(input_signal))*1j)
    else:
        if len(np.shape(input_signal)) == 1:
            noise = math.sqrt(2 * noise_variance) * np.random.randn(len(input_signal))
        else:
            noise = math.sqrt(2 * noise_variance) * np.random.randn(y)

    output_signal = input_signal + noise

    if len(np.shape(input_signal)) == 2:
        plt.subplot(2,1,1)
        plt.title('Modulated Bandpass Signal')
        plt.plot(big_t,input_signal.flatten())
        plt.ylabel('Voltage (V)')
        plt.xlabel('Time (s)')
        plt.subplot(2,1,2)
        plt.title('Modulated Bandpass Signal with Noise')
        plt.plot(big_t,output_signal.flatten())
        plt.ylabel('Voltage (V)')
        plt.xlabel('Time (s)')
        plt.show()
    

    return output_signal
def rayleigh(input_signal, snr_dB):
    if isinstance(input_signal[0], complex):
        h = 1/np.sqrt(2)*(np.random.randn(len(input_signal)) + np.random.randn(len(input_signal))*1j)
        hs =np.multiply(np.abs(h),input_signal)
    else:
        if len(np.shape(input_signal)) == 1:
            h = 1/np.sqrt(2)*(np.random.randn(len(input_signal)))
            hs = np.multiply(h, input_signal)
        else:
            h = 1/np.sqrt(2)*(np.random.randn(y))
            hs = np.multiply(h, input_signal)
    r = awgn(hs, snr_dB)
    return r


def analog2digital(received_signal):
    r = received_signal
    real = []
    imag = []
    half_signal_length = len(r)/2
    half_signal_length = int(half_signal_length)
    for i in range(half_signal_length):
        y = np.sum(r[i])
        real.append(y)
        x = complex(np.sum(r[i + half_signal_length]))
        imag.append(x)
    r = np.vectorize(complex)(real,imag)
    return r
