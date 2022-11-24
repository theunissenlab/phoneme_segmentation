from scipy.io import wavfile

import tables
import numpy as np
from math import *
import matplotlib.pyplot as plt

from .stimulus_utils import load_generic_trfiles
from .interpdata import lanczosinterp2D
from phoneme_segmentation.config import (
        WAV_DIR,
        TRFILE_DIR,
        FEATURES_DIR,
        TRAIN_STORIES,
        TEST_STORIES
        )

ALLSTORIES = sorted(TRAIN_STORIES + TEST_STORIES)

def wav2powspec(start_trim=5,
                end_trim=5,
                fs_check=44100,
                allstories=ALLSTORIES, 
                Rstories=sorted(TRAIN_STORIES), 
                Pstories=TEST_STORIES):
     
    powerspecMatrix_all = {}

    for story in allstories: 
        '''
        step 1: load in wav audio stimulus
        '''
        print(f"load in wav audio data for {story}")

        fs,sound = wavfile.read(f"{WAV_DIR}/{story}.wav")

        if fs != fs_check:
            print (f"sound {story} fs is not {fs_check}, it is {fs}, need resampling")

        '''
        step 2: sound file preprocessing
            i. Convert sound from stereo to mono
            ii. make story zero mean
            iii Using 1sec TR to generate number of TRs

        '''
        print  (f"preprocess: mono and demean sound file for {story}")
        msound = sound.mean(1)-sound.mean()
        TR = 1
        numTRs = ceil((len(msound)/float(fs))/TR)

        '''
        step 3: Make spectrogram
            i. Obtain the beginning and ending indices for a TR
            ii. Run the spectrogram first to get the size of the spectro for one TR
            iii. Pad story with zeros so length(of the story) is multiple of TR length
            iv. Run through all the TR and calculate spectrograms
        '''
        print (f"prep for obtaining spectrogram for {story}")
        sindx = 0 #start index
        eindx = int(floor(TR*1*fs)) # end index

        [spec, fo, to] = makespectrogram(msound[sindx:eindx], fs, 32)
        specMatrix = np.zeros((int(numTRs), spec.shape[0],spec.shape[1]))

        # Pad story with zeros so length(story) is multiple of TR length
        padSound = np.zeros(int(floor(TR*numTRs*fs)))
        padSound[:len(msound)] = msound

        print (f"Run through all the TR and calculate spectrograms for {story}")
        for i in range(int(numTRs)):
            startTime = int(floor(TR*i*fs))
            stopTime = int(floor(TR*(i+1)*fs))
            [spec, fo, to] = makespectrogram(padSound[startTime:stopTime], fs, 32);
            specMatrix[i,:,:] = spec

        print (f"Find the maximum spectrogram in {story}")
        specMax = specMatrix.max()

        '''
        step 4 Make power spectrum
            i. Square the spectrogram and sum over all time for each TR
        '''
        #% Type of power spectrum to generate
        pt_spec = 'log'
        powerspecMatrix = np.zeros((int(numTRs), spec.shape[0]))

        print (f"compute the power spectrum for {story}")
        for tr in range(int(numTRs)):
            powerspecMatrix[tr, :] = makepowerspectrum(specMatrix[tr, :, :],pt_spec, specMax)

        powerspecMatrix_all[story] = powerspecMatrix
   
    '''
    Prep to generate feature matrix
    '''
    trtimes = load_trtimes(allstories)
    powspec_ds_all = powspec_downsample(powerspecMatrix_all, trtimes)

    powspec_Rstim = zscore(np.vstack([powspec_ds_all[story][5+start_trim:-end_trim] for story in Rstories]))
    powspec_Pstim = zscore(np.vstack([powspec_ds_all[story][5+start_trim:-end_trim] for story in Pstories]))

    return powspec_Rstim, powspec_Pstim

def load_trtimes(files_all, trfile_dir=TRFILE_DIR):
    trfiles = load_generic_trfiles(files_all, trfile_dir)
    trtimes = {k: v[0].get_reltriggertimes() for k, v in trfiles.items()}
    return trtimes

def load_datatime(powspec_data, sample_rate=1):
    data_times = {k: np.arange(0, v.shape[0])*sample_rate for k, v in powspec_data.items()}
    return data_times

def powspec_downsample(powspec_data, trtimes, window=3):
    datatime = load_datatime(powspec_data)
    chunsum_all = {k: lanczosinterp2D(powspec_data[k], datatime[k], trtimes[k], window=window) for k, v in powspec_data.items()}

    return chunsum_all


def makespectrogram(sound_in, samprate, fband):

    # Some parameters: these could become flags
    nstd = 6
    twindow = 1000*nstd/(fband*2.0*pi)       # Window length in ms - 6 times the standard dev of the gaussian window
    winLength = np.fix(twindow*samprate/1000.0)  # Window length in number of points
    winLength = int(np.fix(winLength/2)*2)           # Enforce even window length
    increment = int(np.fix(0.001*samprate))           # Sampling rate of spectrogram in number of points - set at 1 kHz

    f_low=25                                  # Lower frequency bounds to get average amplitude in spectrogram
    f_high=15000                            # Upper frequency bound to get average amplitude in spectrogram/ changed from 10000 to 22500
    sil_len=500                             # Amount of silence in ms added at each end of the sound and then subtracted

    soundlen = len(sound_in)
    toPlot = 0                               # toPlot = 1 if you want to plot the spect

    # find the length of the spectrogram and get a time label in ms
    maxlenused = soundlen+np.fix(sil_len*2.0*(samprate/1000.0))
    maxlenint = ceil(maxlenused/increment)*increment

    # Pad the sound with silence
    input_sound = np.zeros(int(maxlenint))
    nzeros = int(np.fix((maxlenint - soundlen)/2))
    input_sound[nzeros:nzeros+soundlen] = sound_in

    # Get the spectrogram
    # Gaussian Spectrum called here to get size of s and fo
    [s, to, fx, bg] = GaussianSpectrum(input_sound, increment, winLength, samprate)
    fstep = fx[2]-fx[1]
    #% low frequency index to get average spectrogram amp
    fl = int(floor(f_low/fstep)+1)
    #% upper frequency index to get average spectrogram amp
    fh = int(ceil(f_high/fstep)+1)
    sabs = abs(s[fl:fh, :])
    fo = fx[fl:fh]


    # Display the spectrogram
    if toPlot == 1:
        plt.figure()
        plt.imshow(sabs)

    return sabs, fo, to


def GaussianSpectrum(input_data, increment, winLength, samprate):
    # %
    # % Gaussian spectrum
    # %     s = GaussianSpectrum(input, increment, winLength, samprate)
    # %     Compute the gaussian spectrogram of an input signal with a gaussian
    # %     window that is given by winLength. The standard deviation of that
    # %     gaussian is 1/6th of winLength.
    # % Each time frame is [winLength]-long and
    # % starts [increment] samples after previous frame's start.
    # % Only zero and the positive frequencies are returned.
    # %   to and fo are the time and frequency for each bin in s and Hz
    # %   pg is a rumming rms.

    # %%%%%%%%%%%%%%%%%%%%%%%
    # % Massage the input
    # %%%%%%%%%%%%%%%%%%%%%%%

    # % Enforce even winLength to have a symmetric window
    if winLength%2 == 1:
        winLength = winLength +1

    winLength = int(winLength)
    # % Make input it into a row vector if it isn't
    # if size(input_data, 1) > 1:
    #     input_data = input_data

    # % Padd the input with zeros
    pinput = np.zeros(len(input_data)+winLength)
    pinput[int(winLength/2):int(winLength/2+len(input_data))] = input_data
    inputLength = len(pinput)

    # % The number of time points in the spectrogram
    # ###%%%think this might be the problem
    frameCount = int(floor((inputLength-winLength)/increment)+1)

    # % The window of the fft
    fftLen = winLength

    # %%%%%%%%%%%%%%%%%%%%%%%%
    # % Gaussian window
    # %%%%%%%%%%%%%%%%%%%%%%%%
    # % Number of standard deviations in one window.
    nstd = 6
    wx2 = (np.arange(winLength)-((winLength+1)/2.0))**2
    wvar = (winLength/nstd)**2
    ws = np.exp(-0.5*(wx2/wvar))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Initialize output "s"
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if fftLen%2 == 1:
        #% winLength is odd
        s = np.zeros((int((fftLen+1)/2+1), frameCount))
    else:
        #% winLength is even
        s = np.zeros((int(fftLen/2+1), frameCount))

    pg = np.zeros(frameCount)

    #% Get FFT from each window
    for i in range(frameCount):
        #%%added fix in here
        start = int(np.fix(i*increment))
        last = int(start + winLength)
        f = np.zeros(fftLen)
        f[:winLength] = np.multiply(ws, pinput[start:last])
        pg[i] = np.std(f[:winLength])

        specslice = np.fft.fft(f)
        #% Take 1/2 because of symmetry
        if fftLen %2 ==1:
            #% winLength is odd
            s[:,i] = specslice[:int((fftLen+1)/2+1)]
        else:
            s[:,i] = specslice[:int(fftLen/2+1)]
        #%s(:,i) = specslice[1:(fftLen/2+1)]

    #% Assign frequency_label
    if fftLen%2 ==1:
        #% winLength is odd
        select = np.array(range(int((fftLen+1)/2)))
    else:
        select = np.array(range(int(fftLen/2+1)))

    fo = select*samprate/fftLen

    #% assign time_label
    to = np.array(range(s.shape[-1]))*increment/samprate

    return s, to, fo, pg

def makepowerspectrum(spec, t, specmax):
#     % Generate power spectrum by taking the sum of the squared or log amplitude
#     % of the spectrogram across time.
#     %
#     % Input
#     % -----
#     % spec : array (f, t)
#     %   Raw spectrogram.
#     %
#     % t : string, 'log', 'square'
#     %   Type of spectrogram.
#     %
#     % specmax : float
#     %   Max of spectrogram over entire story.
#     %
#     % Output
#     % ------
#     % ps : array (1, f)
#     %   Power spectrum.
    DBNOISE = 50;

    if t == 'square':
        mspec = spec ** 2
    elif t == 'log':
        mspec = 20 * np.log10(spec / specmax) + DBNOISE
        mspec[mspec<0.0] = 0.0
    elif t == 'none':
        mspec = spec

    ps = np.sum(mspec, 1)

    return ps

