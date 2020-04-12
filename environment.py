import numpy as np
import math
import random
from mpsk import Mpsk


class Environment:
    def __init__(self, lowest_snr_value, highest_snr_value):
        self.lowest_snr_value = lowest_snr_value
        self.highest_snr_value = highest_snr_value

        
        self.snr1 = lowest_snr_value
        self.snr2 = lowest_snr_value + np.abs((highest_snr_value - lowest_snr_value) / 3)
        self.snr3 = lowest_snr_value + 2 * np.abs((highest_snr_value - lowest_snr_value) / 3)
        self.snr4 = highest_snr_value


        #create 4 signals with either awgn or rayleigh noise, and a predetermined SnR.
        self.signal1 = Mpsk(2,self.snr1, 1000)
        self.signal2 = Mpsk(2,self.snr2, 2000)
        self.signal3 = Mpsk(2,self.snr3, 3000)
        self.signal4 = Mpsk(2,self.snr4, 4000)

        
        self.ber1 = self.signal1.error_rate()
        self.ber2 = self.signal2.error_rate()
        self.ber3 = self.signal3.error_rate()
        self.ber4 = self.signal4.error_rate()

        
        self.obs_space = [self.ber2, self.ber3, self.ber4, self.ber1]
        
    def reset(self):
        
        #re-iniitialize the 4 channels with the SnR of the previous index.
        self.obs_space = np.roll(self.obs_space, 1)
        
        return self.obs_space
    
