import numpy as np
import math
import random



class environment:
    def __init__(self, threshold_value, lowest_snr_value, highest_snr_value):
        self.threshold_value = threshold_value
        self.snr_range = np.arange(lowest_snr_value, highest_snr_value +1)
    def reset(self):
        #create 5 signals with either awgn or rayleigh noise, and a random SnR.
        #re-iniitialize the 5 channels with random SnRs.

        p1,p2,p3,p4,p5 = np.split(self.snr_range,5)
        # for np.split to work, the snr_range must be of a length that is divisible by 5.
        #p1,p2,p3,p4,p5 = random.sample(parts,5)
        

        noise = ['awgn', 'rayleigh']
        signal1 = mpsk(2,random.choice(p1), 1000)
        signal2 = mpsk(2,random.choice(p2), 2000)
        signal3 = mpsk(2,random.choice(p3), 3000)
        signal4 = mpsk(2,random.choice(p4), 4000)
        signal5 = mpsk(2,random.choice(p5), 5000)

        
        ber1 = signal1.error_rate()
        ber2 = signal2.error_rate()
        ber3 = signal3.error_rate()
        ber4 = signal4.error_rate()
        ber5 = signal5.error_rate()

        obs_space = [ber1,ber2,ber3,ber4,ber5]
        
        return obs_space
    
