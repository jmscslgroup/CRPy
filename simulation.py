import numpy as np
import matplotlib.pyplot
import scipy.signal

t = np.linspace(0,0.001,num = 100)

Fc1 = 1000
Fc2 = 2000
Fc3 = 3000
Fc4 = 4000
Fc5 = 5000
Fs = 12000
y1 = 1
y2 = 0
y3 = 0
y4 = 0
y5 = 0
Y = 0
y = 0
x1 = np.cos(2*np.pi*1000*t)

in_p = input("Do you want to enter the first primary user? Y/N:")
if in_p =='Y' or in_p =='y':
    # y1 = amplitude modulation
in_p = input("Do you want to enter the second primary user? Y/N:")
if in_p =='Y' or in_p =='y':
    # y2 = amplitude modulation
in_p = input("Do you want to enter the third primary user? Y/N:")
if in_p =='Y' or in_p =='y':
    # y3 = amplitude modulation
in_p = input("Do you want to enter the fourth primary user? Y/N:")
if in_p =='Y' or in_p =='y':
    # y4 = amplitude modulation
in_p = input("Do you want to enter the fifth primary user? Y/N:")
if in_p =='Y' or in_p =='y':
    # y5 = amplitude modulation
y = y1 + y2 + y3 + y4 + y5

Pxx = scipy.signal.periodogram(y) #this might not be the same as MATLAB Periodogram
Pxx2 = scipy.signal.welch(y) # might also work, need to read the documentation for it


in_p = input("Do you want to enter another primary user? Y/N:")
if in_p =='Y' or in_p =='y':
    tp = 0
    """
    % we’ve obtained five points for all users in the array Pxx which multiplied
    by 10000 should be % above 8000 if there’s no spectral hole. //this just an
    observation which is working so far, the % technical aspects will be
    addressed later in the presentation.
    """
    chek1 = Pxx(25)*10000;
    chek2 = Pxx(46)*10000;
    chek3 = Pxx(62)*10000;
    chek4 = Pxx(89)*10000;
    chek5 = Pxx(105)*10000; #indices will almost certainly have to be changed
    
    if chek1 < 8000:
        print("Assigned to User 1 as it was not present.")
        # y1 = amplitude modulation
    elif chek2 < 8000:
        print("Assigned to User 2 as it was not present.")
        # y2 = amplitude modulation
    elif chek3 < 8000:
        print("Assigned to User 3 as it was not present.")
        # y3 = amplitude modulation
    elif chek4 < 8000:
        print("Assigned to User 4 as it was not present.")
        # y4 = amplitude modulation
    elif chek5 < 8000:
        print("Assigned to User 5 as it was not present.")
        # y5 = amplitude modulation
    else:
        print("All user slots are being used, try again later.")

#rerun PSD code to produce a new PSD Graph
Pxx = scipy.signal.periodogram(y) #this might not be the same as MATLAB Periodogram
Pxx2 = scipy.signal.welch(y) # might also work, need to read the documentation for it

