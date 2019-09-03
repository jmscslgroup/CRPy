import numpy as np
class bpsk:
    def modulate(bit_array, oversampling_factor):
        N = len(bit_array)
        a = 2*bit_array-1
        bit_stream = np.matlib.repmat(a, 1, oversampling_factor)
        bit_stream = bit_stream.flatten()
        t = np.arange(N*oversampling_factor)
        baseband_signal = bit_stream
        return baseband_signal.astype(float), len(t)
    def demodulate(received_bit_array, oversampling_factor):
        x = np.real(received_bit_array)
        x = np.convolve(x, np.ones(oversampling_factor))
        x = x[oversampling_factor-1:len(x):oversampling_factor]
        ak_cap = []
        for i in range(len(x)):
            if x[i] > 0.0:
                ak_cap.append(1)
            else:
                ak_cap.append(0)
        return ak_cap
