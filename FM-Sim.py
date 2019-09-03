# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pylab as pyl

sampling_period = 0.0001

ts = pyl.arange(0, 1, sampling_period)

ym = pyl.sin(2.0 *pyl.pi * 1.0 * ts)

fc = 100.0
mod_fact = 15.0

yc = pyl.sin(2.0 * pyl.pi * (fc + mod_fact * ym) * ts)

pyl.plot(ts, yc)
pyl.show()
