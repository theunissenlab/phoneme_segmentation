# Python Core Dependencies
import numpy as np
import tables

# Theunissen Lab Dependencies
from soundsig.signal import lowpass_filter, highpass_filter
from soundsig.sound import spectrogram, plot_spectrogram

from phoneme_segmentation.config import (
        SIMULATION_BOLD_PATH
        )
# Sample rate
sr = 1/2.0   # 1/TR

print  ("prep to generate simulated BOLD response")
sample_BoldFile = tables.open_file(SIMULATION_BOLD_PATH)
boldData = sample_BoldFile.root.BOLD_sample.read()

meanBold = np.mean(boldData[25:])
rmsBold = np.std(boldData[25:])

def mod1_generation(featureR, featureP, selection, featureR_orig=False):
	'''
		generate one voxel's BOLD response encoding one feature 
	'''
    if False in featureR_orig:
        meanR = np.mean(featureR, axis=0)
    else:
        meanR = np.mean(featureR_orig, axis=0)
        print ("shape of meanR is %s"%(meanR.shape))
    
    indSort = np.argsort(meanR)

    h = np.zeros((featureR.shape[-1], 3))
    h[indSort[selection],0] = 0.5
    h[indSort[selection],1] = 1.0
    h[indSort[selection],2] = -0.5

    boldMod1R = np.zeros(featureR.shape[0])
    boldMod1R[1:] = boldMod1R[1:] + np.dot(featureR[0:-1],h[:,0])
    boldMod1R[2:] = boldMod1R[2:] + np.dot(featureR[0:-2],h[:,1])
    boldMod1R[3:] = boldMod1R[3:] + np.dot(featureR[0:-3],h[:,2])

    boldMod1P = np.zeros(featureP.shape[0])
    boldMod1P[1:] = boldMod1P[1:] + np.dot(featureP[0:-1],h[:,0])
    boldMod1P[2:] = boldMod1P[2:] + np.dot(featureP[0:-2],h[:,1])
    boldMod1P[3:] = boldMod1P[3:] + np.dot(featureP[0:-3],h[:,2])

    # Save this signal for model 3
    sig1R = boldMod1R
    sig1P = boldMod1P

    # Low pass filter
    boldMod1R = lowpass_filter(sig1R, sr, 0.1, filter_order=4)
    boldMod1P = lowpass_filter(sig1P, sr, 0.1, filter_order=4)

    # Add noise with a given SNR
    SNR = 2.0
    sdB1R = np.std(boldMod1R)
    sdB1P = np.std(boldMod1P)

    noise = np.random.normal(loc=0.0, scale=sdB1R/np.sqrt(SNR), size=boldMod1R.shape)
    boldMod1R = boldMod1R + noise

    noise = np.random.normal(loc=0.0, scale=sdB1P/np.sqrt(SNR), size=boldMod1P.shape)
    boldMod1P = boldMod1P + noise

    # Match rms and mean of example Bold
    meanB1R = np.mean(boldMod1R)
    sdB1R = np.std(boldMod1R)
    boldMod1R = (boldMod1R-meanB1R)*rmsBold/sdB1R + meanBold

    meanB1P = np.mean(boldMod1P)
    sdB1P = np.std(boldMod1P)
    boldMod1P = (boldMod1P-meanB1P)*rmsBold/sdB1P + meanBold

    return sig1R, sig1P, boldMod1R, boldMod1P

def mod2_generation(sig1R, sig1P, sig2R, sig2P):
        '''
                generate one voxel's BOLD response encoding two features 
        '''

    boldMod3R = sig1R + sig2R
    boldMod3P = sig1P + sig2P

    ## Process model Bold signal to make it more realistic
    # Low pass filter
    boldMod3R = lowpass_filter(boldMod3R, sr , 0.1, filter_order=4)
    boldMod3P = lowpass_filter(boldMod3P, sr , 0.1, filter_order=4)

    # Add noise with a given SNR
    SNR = 2.0
    sdB3R = np.std(boldMod3R)
    noise = np.random.normal(loc=0.0, scale=sdB3R/np.sqrt(SNR), size=boldMod3R.shape)
    boldMod3R = boldMod3R + noise

    sdB3P = np.std(boldMod3P)
    noise = np.random.normal(loc=0.0, scale=sdB3P/np.sqrt(SNR), size=boldMod3P.shape)
    boldMod3P = boldMod3P + noise

    # Match rms and mean of example Bold
    meanB3R = np.mean(boldMod3R)
    sdB3R = np.std(boldMod3R)
    boldMod3R = (boldMod3R-meanB3R)*rmsBold/sdB3R + meanBold

    meanB3P = np.mean(boldMod3P)
    sdB3P = np.std(boldMod3P)
    boldMod3P = (boldMod3P-meanB3P)*rmsBold/sdB3P + meanBold

    return boldMod3R, boldMod3P
