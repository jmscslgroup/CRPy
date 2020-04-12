"""
awgn function provided from https://github.com/veeresht/CommPy/blob/master/commpy/channels.py, the rest was made from scratch

A MPSK (2,4 and 8) modulation and demodulation scheme that takes an array of bits (must be in string format like ['01','11','10'] instead of [101, 11, 10]) as input.
self.send() and self.receive() are combinations of the other functions in this class in order to make the entire process easier.

Author: Brandon Dominique
Email: dominiquebdmnq@aol.com
Date:8/2/19
"""

import numpy as np
import math
import matplotlib.pyplot as plt
class Mpsk:
    
    def __init__(self, mod_number, snr_dB, carrier_frequency):
        self.mod_number = mod_number
        self.snr_dB = snr_dB
        self.carrier_frequency = carrier_frequency
    def modulate(self, bit_array):
        """
        Modulation for an array of bits.
        Parameters
        ----------
        bit_array : array of a string of bits (Ex. ['0','1','1'] or ['111','010']
        
        mod_number: an integer that specifies the modualtion scheme you want to use.
        as of now mpsk.py supports 2,4 and 8 PSK modulation.
        Returns
        -------
        modulated_bits : array of bits mapped to a point on the unit circle(divided by sqrt(2)) in the format [(a+bj)]
        reference_constellation : entire reference constellation for the given modulation scheme M (used later to demodulate the signal).
        """
        mod_number = self.mod_number
        ref_i = []
        ref_q = []
        modulated_bits = []

        
        for i in range(mod_number):
            ref_i.append(1/np.sqrt(2)*np.cos(((i))/mod_number*2*np.pi))
            ref_q.append(1/np.sqrt(2)*np.sin(((i))/mod_number*2*np.pi))
            """
            np.sin(np.pi) and np.cos(np.pi/2) don't produce the values they're
            expected to because np.pi is a irrational number; instead of
            np.sin(np.pi) equalling 0, it will equal 1.2246467991473532e-16
            and np.cos(np.pi/2) will equal 6.123233995736766e-17. The next few if
            statements are to make sure that the correct value is appended to the
            array if this problem occurs.
            """
            if -0.01<= ref_q[i] <= 0.01:
                ref_q[i] = 0.0
            if -0.01<= ref_i[i] <= 0.01:
                ref_i[i] = 0.0
            if -0.51<= ref_q[i] <= -0.49:
                ref_q[i] = -0.5
            if -0.51<= ref_i[i] <= -0.49:
                ref_i[i] = -0.5
            if 0.49<= ref_q[i] <= 0.51:
                ref_q[i] = 0.5
            if 0.49<= ref_i[i] <= 0.51:
                ref_i[i] = 0.5
        ref = np.vectorize(complex)(ref_i, ref_q)
        reference_constellation = ref
        
        for i in range(len(bit_array)):
            if mod_number == 2: #BPSK Modulation
                if bit_array[i] == '0':
                    modulated_bits.append(ref[1])
                elif bit_array[i] == '1':
                    modulated_bits.append(ref[0])
            elif mod_number == 4: #QPSK Modulation
                if bit_array[i] == '00':
                    modulated_bits.append(ref[0])
                elif bit_array[i] == '01':
                    modulated_bits.append(ref[1])
                elif bit_array[i] == '10':
                    modulated_bits.append(ref[2])
                elif bit_array[i] == '11':
                    modulated_bits.append(ref[3])
            elif mod_number == 8: #8PSK Modulation
                if bit_array[i] == '000':
                    modulated_bits.append(ref[0])
                elif bit_array[i] == '001':
                    modulated_bits.append(ref[1])
                elif bit_array[i] == '010':
                    modulated_bits.append(ref[2])
                elif bit_array[i] == '011':
                    modulated_bits.append(ref[3])
                elif bit_array[i] == '100':
                    modulated_bits.append(ref[4])
                elif bit_array[i] == '101':
                    modulated_bits.append(ref[5])
                elif bit_array[i] == '110':
                    modulated_bits.append(ref[6])
                elif bit_array[i] == '111':
                    modulated_bits.append(ref[7])
        self.reference_constellation = reference_constellation
        return modulated_bits
        
    
    def digital2analog(self, mod):
        """
        Convert a digital signal into an analog one so that it can be transmitted.
        Parameters
        ----------
        bit_array : array of a string of bits (Ex. ['0','1','1'] or ['111','010']

        Returns
        -------
        signal : an array that is half cos, half sin.
        If you modulated N bits in modulation(), this will return a 2N*100 size array.
        Each N has its real portion multiplied by np.cos, and its imaginary portion
        multiplied by np.sin.
        
        """

        carrier_frequency = self.carrier_frequency
        fc = carrier_frequency
        start = 0
        stop = 0.1
        number_of_steps = 100
        
        time_for_individual_signal = np.linspace(start,stop,num = number_of_steps)
        t = time_for_individual_signal

        
        signal = []
        inphase = np.reshape(np.real(mod),(len(mod),1))*np.reshape(np.cos(2*np.pi*fc*t),(1,number_of_steps))
        quadrature = np.reshape(np.imag(mod),(len(mod),1))*np.reshape(np.round(np.sin(2*np.pi*fc*t),3),(1,number_of_steps))
        signal = np.concatenate((inphase,quadrature),axis = 0)
        return signal









    
    def awgn(self, input_signal, rate=1.0):
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

        snr_dB = self.snr_dB
        if len(np.shape(input_signal)) == 2:
            x,y = np.shape(input_signal)
            

        avg_energy = np.sum(np.abs(input_signal) * np.abs(input_signal)) / len(input_signal)
        snr_linear = 10 ** (snr_dB / 10.0)
        noise_variance = avg_energy / (2 * rate * snr_linear)

        if isinstance(input_signal[0], complex):
            noise = (math.sqrt(noise_variance) * np.random.randn(len(input_signal))) + (math.sqrt(noise_variance) * np.random.randn(len(input_signal))*1j)
        else:
            if len(np.shape(input_signal)) == 1:
                noise = math.sqrt(2 * noise_variance) * np.random.uniform(-.2,.2,size = len(input_signal))
            else:
                """
                This else statement is meant to run after you pass asignal through modulate()
                and digital2analog(). It is meant to assign a unique amount of noise to each N
                of your 2N*100 array, but it instead generates a random 1*100 array and applies this
                to each N, meaning that every bit doesn't get its own unique amount of noise. I'm working on fixing
                this at the moment.

                Maybe use .flatten(), then add noise to every bit, then reshape it to 2N*100 again?
                """
                noise = math.sqrt(2 * noise_variance) * np.random.uniform(-.2,.2,size = y)

        output_signal = input_signal + noise

        return output_signal

    
    def rayleigh(self, input_signal):
        """
        Rayleigh Channel.
        Parameters
        ----------
        input_signal : 1D ndarray of floats
            Input signal to the channel.
        snr_dB : float
            Output SNR required in dB.
        Returns
        -------
        r : 1D ndarray of floats
            Output signal from the channel with the specified SNR.
            Rayleigh takes a signal s and creates the faded signal H*s,
            then calls awgn() to create H*s + noise.
        """

        snr_dB = self.snr_dB
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
        r = self.awgn(hs, snr_dB)
        return r
    
    def analog2digital(self, received_signal):
        """
        Get the received signal turned back into complex form.
        Parameters
        ----------
        received_signal : noisy array that is half cos, half sin. The size is
        N*100 where N is the number of bits that were transmitted in your original
        array.
        
        Returns
        -------
        r : array of bits in the format [(a+bj)]
        """
        r = received_signal
        real = []
        imag = []
        half_signal_length = len(r)/2
        half_signal_length = int(half_signal_length)
        for i in range(half_signal_length):
            y = np.sum(r[i])
            real.append(y)
            x = complex(np.max(r[i + half_signal_length]))
            imag.append(x)
        r = np.vectorize(complex)(real,imag)
        return r



    
    def demodulate(self, received_bits):
        """
        Demodulation by Computing the Euclidean distance of each index in
        received_bits. Basically, calculate the distance of each
        index of received_bits to the original positions on the MPSK
        constellation; whichever constellation has the shortest distance to the
        index is determined to have that value.

        recieved_bits should be a 1-D array of complex bits.

        output is an array of the bit with the closest value to the received bit.
        """
        reference_constellation = self.reference_constellation
        mod_number = self.mod_number
        m = len(received_bits)
        n = len(reference_constellation)
        
        x = received_bits
        x = np.column_stack((np.real(x),np.imag(x)))
        X = np.sum(np.multiply(x,x), axis=1)
        X = X.reshape((m,1))


        y = reference_constellation
        y = np.column_stack((np.real(y),np.imag(y)))
        Y = np.sum(np.multiply(y,y), axis=1)
        Y=Y.reshape((1,n))

        X_copy = X
        for i in range(n-1):
            x_copy =np.copy(X_copy)
            X = np.hstack((X,x_copy))

        Y_copy = Y
        for i in range(m-1):
            y_copy =np.copy(Y_copy)
            Y = np.vstack((Y,y_copy))

        d = X + Y -2*np.matmul(x,np.transpose(y))
        ideal_points = np.argmin(d, axis=1)
        demodulated_points = [y[i] for i in ideal_points]
        demodulated_points = np.round(demodulated_points, 4)
        demod_signal = []
        
        
        for i in range(len(demodulated_points)):
            if mod_number == 2:
                if demodulated_points[i][0] == 0.7071 and demodulated_points[i][1] == 0.0:
                    demod_signal.append('1')
                elif demodulated_points[i][0] == -0.7071 and demodulated_points[i][1] == 0.0:
                    demod_signal.append('0')
                else:
                    demod_signal.append('(theres supposed to be a bit here but there was some type of error)')
            elif mod_number == 4:
                if demodulated_points[i][0] == 0.7071 and demodulated_points[i][1] == 0.0:
                    demod_signal.append('00')
                elif demodulated_points[i][0] == 0.0 and demodulated_points[i][1] == 0.7071:
                    demod_signal.append('01')
                elif demodulated_points[i][0] == -0.7071 and demodulated_points[i][1] == 0.0:
                    demod_signal.append('10')
                elif demodulated_points[i][0] == 0.0 and demodulated_points[i][1] == -0.7071:
                    demod_signal.append('11')
                else:
                    demod_signal.append('(theres supposed to be a bit here but there was some type of error)')
            elif mod_number == 8:
                if demodulated_points[i][0] == 0.7071 and demodulated_points[i][1] == 0.0:
                    demod_signal.append('000')
                elif demodulated_points[i][0] == 0.5 and demodulated_points[i][1] == 0.5:
                    demod_signal.append('001')
                elif demodulated_points[i][0] == 0.0 and demodulated_points[i][1] == 0.7071:
                    demod_signal.append('010')
                elif demodulated_points[i][0] == -0.5 and demodulated_points[i][1] == 0.5:
                    demod_signal.append('011')
                elif demodulated_points[i][0] == -0.7071 and demodulated_points[i][1] == 0.0:
                    demod_signal.append('100')
                elif demodulated_points[i][0] == -0.5 and demodulated_points[i][1] == -0.5:
                    demod_signal.append('101')
                elif demodulated_points[i][0] == 0.0 and demodulated_points[i][1] == -0.7071:
                    demod_signal.append('110')
                elif demodulated_points[i][0] == 0.5 and demodulated_points[i][1] == -0.5:
                    demod_signal.append('111')
                else:
                    demod_signal.append('(theres supposed to be a bit here but there was some type of error')
        return demod_signal
    
    def send(self, bit_array, noise = None, graph = 'no'):
        mod = self.modulate(bit_array)
        #print("Modulated Signal: ", mod)
        signal = self.digital2analog(mod)
        if noise == 'awgn': #None represents no noise by default
            signal_with_noise = self.awgn(signal)
        elif noise == 'rayleigh':
            signal_with_noise = self.rayleigh(signal)
        else:
            signal_with_noise = signal

        if graph == 'yes':
            # generating a graph to visualize each bit that was modulated and multiplied by
            # either cos or sin
            start = 0
            stop = 0.1
            number_of_steps = 100
        
            time_for_individual_signal = np.linspace(start,stop,num = number_of_steps)
            t = time_for_individual_signal
    
            time_for_entire_modulated_signal = np.linspace(start,stop*2*len(mod),num = number_of_steps*2*len(mod))
            big_t = time_for_entire_modulated_signal

            time_for_part_of_signal = np.linspace(start,stop*len(mod),num = number_of_steps*len(mod))
            half_t = time_for_part_of_signal

            inphase = np.reshape(np.real(mod),(len(mod),1))*np.reshape(np.cos(2*np.pi*self.carrier_frequency*t),(1,number_of_steps))
            quadrature = np.reshape(np.imag(mod),(len(mod),1))*np.reshape(np.sin(2*np.pi*self.carrier_frequency*t),(1,number_of_steps))
        







            plt.subplot(2,2,1)
            plt.title('Modulated Bandpass Signal (Real Part)')
            plt.plot(half_t,inphase.flatten())
            plt.ylabel('Voltage (V)')
            plt.xlabel('Time (s)')
            plt.subplot(2,2,2)
            plt.title('Modulated Bandpass Signal (Imaginary Part)')
            plt.plot(half_t,quadrature.flatten())
            plt.ylabel('Voltage (V)')
            plt.xlabel('Time (s)')
            plt.subplot(2,2,3)
            plt.title('Entire Modulated Bandpass Signal')
            plt.plot(big_t,signal.flatten())
            plt.ylabel('Voltage (V)')
            plt.xlabel('Time (s)')
            plt.subplot(2,2,4)
            plt.title('Modulated Bandpass Signal with Noise')
            plt.plot(big_t,signal_with_noise.flatten())
            plt.ylabel('Voltage (V)')
            plt.xlabel('Time (s)')
            plt.show()
        
        return signal_with_noise

    def receive(self, received_signal):
        complex_array = self.analog2digital(received_signal)
        output_array = self.demodulate(complex_array)
        print("Output of Demodulated Signal: ",output_array)
        return output_array

    def error_rate(self, noise = 'awgn'):
        test = np.random.rand(50000)
        ak = []
        for i in range(len(test)):
            if test[i] > 0.5:
                ak.append('1')
            else:
                ak.append('0') 

        errors = []
        EbNodB = self.snr_dB

        if noise == 'rayleigh':
            signal = Mpsk(2,EbNodB,1000)
        else:
            signal = Mpsk(2,EbNodB,1000)
        modulated_bits = signal.modulate(ak)

        if noise == 'rayleigh':
            bits_with_noise = signal.rayleigh(modulated_bits)
        else:
            bits_with_noise = signal.awgn(modulated_bits)
        
        demodulated_bits = signal.demodulate(bits_with_noise)
        for i in range(len(demodulated_bits)):
                if demodulated_bits[i] != ak[i]:
                        errors.append('1')
        ber = 1.0 * len(errors) / 50000
        
        print("Signal to Noise Ratio(EbNodB):", EbNodB)
        print("Error bits:", len(errors))
        print("Error probability:", ber)


        return ber 
