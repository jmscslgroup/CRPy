#https://github.com/veeresht/CommPy/blob/master/commpy/modulation.py
# Authors: Veeresh Taranalli <veeresht@gmail.com> & Bastien Trotobas <bastien.trotobas@gmail.com>
# License: BSD 3-Clause

"""
==================================================
Modulation Demodulation (:mod:`commpy.modulation`)
==================================================
.. autosummary::
   :toctree: generated/
   PSKModem             -- Phase Shift Keying (PSK) Modem.
   QAMModem             -- Quadrature Amplitude Modulation (QAM) Modem.
   ofdm_tx              -- OFDM Transmit Signal Generation
   ofdm_rx              -- OFDM Receive Signal Processing
   mimo_ml              -- MIMO Maximum Likelihood (ML) Detection.
   kbest                -- MIMO K-best Schnorr-Euchner Detection.
   bit_lvl_repr         -- Bit level representation
"""
from itertools import product

import matplotlib.pyplot as plt
from numpy import arange, array, zeros, pi, cos, sin, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron
from numpy.fft import fft, ifft
from numpy.linalg import qr, norm
from commpy.utilities import bitarray2dec, dec2bitarray

__all__ = ['PSKModem', 'QAMModem', 'ofdm_tx', 'ofdm_rx', 'mimo_ml', 'kbest', 'bit_lvl_repr']



class Modem:
    def modulate(self, input_bits):
        """ Modulate (map) an array of bits to constellation symbols.
        Parameters
        ----------
        input_bits : 1D ndarray of ints
            Inputs bits to be modulated (mapped).
        Returns
        -------
        baseband_symbols : 1D ndarray of complex floats
            Modulated complex symbols.
        """
        mapfunc = vectorize(lambda i:
                            self.constellation[bitarray2dec(input_bits[i:i + self.num_bits_symbol])])

        baseband_symbols = mapfunc(arange(0, len(input_bits), self.num_bits_symbol))

        return baseband_symbols

    def demodulate(self, input_symbols, demod_type, noise_var=0):
        """ Demodulate (map) a set of constellation symbols to corresponding bits.
        Parameters
        ----------
        input_symbols : 1D ndarray of complex floats
            Input symbols to be demodulated.
        demod_type : string
            'hard' for hard decision output (bits)
            'soft' for soft decision output (LLRs)
        noise_var : float
            AWGN variance. Needs to be specified only if demod_type is 'soft'
        Returns
        -------
        demod_bits : 1D ndarray of ints
            Corresponding demodulated bits.
        """
        if demod_type == 'hard':
            index_list = map(lambda i: argmin(abs(input_symbols[i] - self.constellation)),
                             range(0, len(input_symbols)))
            demod_bits = hstack(map(lambda i: dec2bitarray(i, self.num_bits_symbol),
                                    index_list))
        elif demod_type == 'soft':
            demod_bits = zeros(len(input_symbols) * self.num_bits_symbol)
            for i in arange(len(input_symbols)):
                current_symbol = input_symbols[i]
                for bit_index in arange(self.num_bits_symbol):
                    llr_num = 0
                    llr_den = 0
                    for const_index in self.symbol_mapping:
                        if (const_index >> bit_index) & 1:
                            llr_num = llr_num + exp(
                                (-abs(current_symbol - self.constellation[const_index]) ** 2) / noise_var)
                        else:
                            llr_den = llr_den + exp(
                                (-abs(current_symbol - self.constellation[const_index]) ** 2) / noise_var)
                    demod_bits[i * self.num_bits_symbol + self.num_bits_symbol - 1 - bit_index] = log(llr_num / llr_den)
        else:
            pass
            # throw an error

        return demod_bits
