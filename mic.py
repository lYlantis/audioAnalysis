#! python3


import matplotlib
matplotlib.use('Agg')                           #Allows you to use matplotlib without having a display
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import math
import time
from scipy.io import wavfile
import pyaudio
import logging
import numpy as np
from scipy.fftpack import fft, fftfreq
import wave
import random
import math
import struct
#import audioFeatureExtraction

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

 
def getAudio():
    '''
    Records audio for X seconds, based on parameters in heading
    Parameters
    ----------
    Nothing
    
    Returns
    -------
    frames : numpy array with audio data
    '''
    CHUNK = 512  # Changing the recording seconds might require a change to this value
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5  # Use .125 in real setting. 5 seconds was used for testing audio validity
    CHUNKS_TO_RECORD = int(RATE / CHUNK * RECORD_SECONDS)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    logging.debug("* Recording audio...")

    frames = np.empty((CHUNKS_TO_RECORD * CHUNK), dtype="int16")

    for i in range(0, CHUNKS_TO_RECORD):
        audioString = stream.read(CHUNK)
        frames[i * CHUNK:(i + 1) * CHUNK] = np.fromstring(audioString, dtype="int16")

    logging.debug("* done recording\n")

    logging.debug("closing stream")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return frames

def makeWav(frames, OutputName='output.wav'):
    '''
    Makes a .wav file with the audio data in frames
    
    Parameters
    ----------
    frames : numpy array of audio data
    OutputName : string value for the output name

    Returns
    -------
    Nothing
    '''
    logging.debug("making you a wav file\n")
    waveFile = wave.open(OutputName, 'wb')
    waveFile.setnchannels(CHANNELS) 
    waveFile.setsampwidth(p.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    logging.debug('Done making "%s"' % (OutputName))

def freqMaker(frequency=440.0, OutputName="wavTester.wav"):
    '''
    Makes a .wav file for a specific tone
    
    Parameters
    ----------
    frequency : frequency of the tone
    OutputName : string value for the output name

    Returns
    -------
    Nothing
    
    src:  http://stackoverflow.com/questions/3637350/how-to-write-stereo-wav-files-in-python
    src:  http://www.sonicspot.com/guide/wavefiles.html
    '''
    freq = frequency            #Generates a 440 hz tone if no other frequency is specified
    data_size = 32768           #2^15
    fname = OutputName
    frate = RATE
    amp = 64000.0
    nchannels = CHANNELS          
    sampwidth = 2
    framerate = int(frate)
    nframes = data_size
    comptype = "NONE"
    compname = "not compressed"
    data = [math.sin(2 * math.pi * freq * (x / frate))
            for x in range(data_size)]
    wav_file = wave.open(fname, 'w')
    wav_file.setparams(
        (nchannels, sampwidth, framerate, nframes, comptype, compname))
    for v in data:
        wav_file.writeframes(struct.pack('h', int(v * amp / 2)))
    wav_file.close()

def plotTimeAndFft(frames, sampFreq=44100):
    '''
    Takes a time signal as input, performs a fft, and plots both signals together.
    
    Parameters
    ----------
    frames : numpy array of audio data
    sampFreq : frequency the audio data was sampled at

    Returns
    -------
    Nothing
    '''
    lenData=float(len(frames))
        
    #plotting the sound in time
    timeArray = np.linspace(0, lenData,lenData)
    timeArray = timeArray / sampFreq
    timeArray = timeArray * 1000  #scale to milliseconds
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(timeArray, data, color='k')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.plot(timeArray, data, color='k')

    signalFFT = fft(data)

    nUniquePts = int(math.ceil((lendata+1)/2.0))

    signalFFT = signalFFT[0:nUniquePts]
    signalFFT = abs(signalFFT)             #By taking the absolute value of the fft we get information about the magnitude of the frequency components.
    signalFFT = signalFFT / float(lendata) # scale by the number of points so that
                                           # the magnitude does not depend on the length 
                                           # of the signal or on its sampling frequency
                                           
    signalFFT = signalFFT**2  # square it to get the power 

    # multiply by two (see https://web.archive.org/web/20120615002031/http://www.mathworks.com/support/tech-notes/1700/1702.html for details)
    # odd nfft excludes Nyquist point
    
    if lendata % 2 > 0: # we've got odd number of points fft
        signalFFT[1:len(signalFFT)] = signalFFT[1:len(signalFFT)] * 2
    else:
        signalFFT[1:len(signalFFT) -1] = signalFFT[1:len(signalFFT) - 1] * 2 # we've got even number of points fft

    #Plot the frequency spectrum of the audio
    freqArray = np.arange(0, nUniquePts, 1.0) * (sampFreq / lendata);
    plt.subplot(212)
    plt.plot(freqArray/1000, 10*np.log10(signalFFT), color='k')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.tight_layout()
    plt.text(10 ,0, 'Peak Freq: %s' % str(freqArray[idx_max]))
    plt.savefig('sig+fft.png', bbox_inches='tight')
    plt.close()

def getDb(frames):
    '''
    Gets the decibel value from a numpy array of audio data
    
    Parameters
    ----------
    frames : numpy array of audio data

    Returns
    -------
    dB : the average decibel value over the length of the frames
    '''
    frames_copy = frames.astype(np.float)
    dat = frames_copy / 32768.0
    magSq = np.sum(dat ** 2.0) / len(dat)
    dB = 10.0 * math.log(magSq,10.0)
    return dB

def getRms(frames):
    '''
    Gets the normalized RMS value from a numpy array of audio data
    
    Parameters
    ----------
    frames : numpy array of audio data

    Returns
    -------
    rms : the normalized average RMS value over the length of the frames
    '''
    frames = frames / (2.0 ** 15)  # convert it to floating point between -1 and 1
    rms = np.sqrt(np.mean(np.square(frames)))
    return rms

def getZeroCrossing(frames):
    '''
    ** Note: Needs work to determine suitability for project needs **
    
    Gets the Zero Crossing Rate value from a numpy array of audio data
    
    Parameters
    ----------
    frames : numpy array of audio data

    Returns
    -------
    zcr : the average ZCR value over the length of the frames (???)

    src: pyAudioAnalysis library
    motivation: https://pdfs.semanticscholar.org/cd34/eaa0029ba9f6cfc4a1ed64722e05a476079f.pdf
    '''
    count = len(frames)
    countZ = np.sum(np.abs(np.diff(np.sign(frames)))) / 2
    zcr = (np.float64(countZ) / np.float64(count - 1.0))
    return zcr

def getShortTermEnergy(frames):
    '''
    ** Note: Needs work to determine suitability for project needs **
    
    Calculates the short-term energy of an audio frame. The energy value is
    normalized using the length of the frame to make it independent of said
    quantity.
    
    src/motivation: http://bastian.rieck.ru/blog/posts/2014/simple_experiments_speech_detection/
    '''
    
    return np.sum( [ np.abs(x)**2 for x in frames ] ) / len(frames)

def getLogarithmicEnergy(frames):
    '''
    ** Note: Needs work to determine suitability for project needs **
    
    returns the log of the short term energy

    src/motivation: http://airccse.org/journal/ijsc/papers/4213ijsc03.pdf
    '''
    return 20*np.log(getShortTermEnergy(frames))

def getShortTimeFourierTransform(frames, fs=44100):
    '''
    ** Note: Needs work to determine suitability for project needs **
    
    Parameters
    ----------
    frames : a numpy array containing the signal to be processed
    fs : a scalar which is the sampling frequency of the data

    Returns
    -------
    result : ???
    
    src: https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
    motivation: http://airccse.org/journal/ijsc/papers/4213ijsc03.pdf
    '''
    fft_size=512
    overlap_fac=.5
    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(frames) / np.float32(hop_size)))
    t_max = len(frames) / np.float32(fs)

    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

    proc = np.concatenate((frames, np.zeros(pad_end_size)))  # the data to process
    result = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the result

    for i in range(total_segments):  # for each segment
        current_hop = hop_size * i  # figure out the current segment offset
        segment = proc[current_hop:current_hop + fft_size]  # get the current segment
        windowed = segment * window  # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size  # take the Fourier Transform and scale by the number of samples
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        result[i, :] = autopower[:fft_size]  # append to the results array

    result = 20 * np.log10(result)  # scale to db
    result = np.clip(result, -40, 200)  # clip values
    return result

def detectStreamingVolumeEvent():
    '''
    Records a stream of data and decides if an event has occured based on threshold.
    '''
    RMS_GUESS = 0.05
    
    frames = getAudio()
    dB = getDb(frames)
    rmsDB = getRms(dB)
    rms = getRms(frames)
    
    if rms > 2.5*RMS_GUESS:
        logging.info('\n Event Detected')
        logging.info("dB: %s" % dB)
        logging.info("rms (dB): %s" % rmsDB)
        logging.info("rms of frames: %s" % rms)

def getPeak(frames, rate=44100):
    '''
    Finds the peak in the coefficients
    
    Parameters
    ----------
    frames : a numpy array containing the signal to be processed
    rate : a scalar which is the sampling frequency of the data

    Returns
    -------
    peak_freq_in_hertz : The peak frequency value after fft of the signal
    '''
    signalFFT = fft(frames)
    freqs = fftfreq(len(signalFFT))
    idx = np.argmax(np.abs(signalFFT))
    freq = freqs[idx]
    peakFreqInHertz = abs(freq * rate)
    return peakFreqInHertz

def getRecordedFrames(fileName, Channels=1):
    '''
    Gets a signal to analyze from a wav file
    
    Parameters
    ----------
    fileName : the filname containing the signal
    Channels : 1 = mono, 2 = stereo; currently only mono analysis

    Returns
    -------
    frames : an array containg the signal
    sampFreq : the frequency that the signal was sampled
    '''
    sampFreq, frames = wavfile.read(fileName)
    lenData=float(len(frames))

    #turn a stereo recording into mono
    if Channels>1:
        frames = frames[:,0]

    ##uncomment next line if you want an int16 data to be converted to floating values between -1 and 1
    #frames = frames / (2.0**15)                    #convert it to floating point between -1 and 1

    logging.info('duration in seconds: "%s"' % (lenData / sampFreq))  # gives the duration of the sample
    logging.info('duration in minutes: "%s"' % (lenData / sampFreq/60.60))  # gives the duration of the sample
    return (frames, sampFreq)

def rmsAnalysisNoOverlapping():
    '''
    Basic testing for volume event detection - non overlapping windows, events temporally linked, no hamming window
    '''

    MULTIPLIER = 1.75
    WINDOW_SIZE = .5
    eventCount=0

    #First file is for RMS refernece, Second file is for event analyszis
    backFrames, backSamp = getRecordedFrames('C:\\Users\\...')
    frames, sampFreq = getRecordedFrames('C:\\Users\\...')

    #Information on the audio file to analyze
    lenData = float(len(frames))                                    #Samples
    howManyChunks = math.floor((lenData/sampFreq)/WINDOW_SIZE)      #(lenData/sampFreq)=Duration of recording. (Duration/WindowSize)=# of windows
    logging.info('breaking it into: "%s" Chunks' % (howManyChunks))
    whereToBreak = int(lenData/howManyChunks)                       #Size of Windows in samples
    logging.info('breaking frames every: "%s" Samples' % (whereToBreak))

    #RMS threshold
    longtermAvgVol = getDb(backFrames)
    longtermAvgRms = getRms(backFrames)
    #longtermAvgRms = .0434696

    #Plot the audio signal and RMS threshold
    plt.clf()
    #timeArray = np.linspace((i/sampFreq),(j/sampFreq),len(howManyChunks*whereToBreak))
    f,ax = plt.subplots(2, figsize=(12,10), dpi=100,sharex=True)
    timeArray = np.linspace(0, (lenData/ sampFreq), lenData)
    ax[0].plot(timeArray, frames)
    ax[0].set_title('Audio Signal')
    ax[0].set_ylabel('Amplitude')
    horizLine = np.array([MULTIPLIER*longtermAvgRms for i in range(len(timeArray))])    #Plot a horizontal like for the volume Threshold
    logging.info('The threshold level is: "%s"' % (MULTIPLIER*longtermAvgRms))
    ax[1].plot(timeArray, horizLine, label='Detection Level: '+str(MULTIPLIER)+'x')

    #window = np.hamming(whereToBreak)

    seg_length = whereToBreak
    i, j = 0, seg_length
    oldi=0
    oldj=0
    frames_chunk=[]
    #Now break up the audio and do analysis
    while j <= (howManyChunks*whereToBreak):
        framesChunk = frames[i: j]
        #framesChunk = framesChunk*(window)
        t = ((i/sampFreq) + (j/sampFreq)) / 2
        decibel = getDb(framesChunk)
        #logging.info('t: "%s"' % t)
        #logging.info('i time: "%s", j time: "%s"' % ((i/sampFreq),(j/sampFreq)))
        
        ## If we were building up the longterm RMS
        #longtermAvgVol = (longtermAvgVol + decibel) / 2
        
        rms = getRms(framesChunk)
        ax[1].plot(t, rms, marker='.', color='k')
        #peakFreq = getPeak(frames_chunk, sampFreq)
        
        if rms>(MULTIPLIER*longtermAvgRms):
            logging.info('i: "%s", oldj: "%s"' % (i, oldj))
            if i-WINDOW_SIZE<oldj:
                #events are the same if within a window size of each other
                plt.plot(t, rms, marker='o', color='r')
                a=[[t, rms],[oldx,oldy]]
                plt.plot(*zip(*a), marker='o', linestyle='-',color='r')
                #logging.info('t: "%s", RMS: "%s", oldx: "%s", oldy: "%s"' %(t, rms, oldx, oldy)) 
            else:
                eventCount += 1
                print('\n')
                logging.info('Event Detected ')
                logging.info('i time: "%s", j time: "%s"' % ((i/sampFreq),(j/sampFreq)))
                #logging.info('Chunk #: "%s", Starts at: "%s"'% (i, timeVal))
                logging.info('rms: "%s", MULTIPLIER * longtermRMS: "%s"\n' % (rms, abs(MULTIPLIER*longtermAvgRms)))
                plt.plot(t, rms, marker='o', color='r')
                oldx=t
                oldy=rms
            oldi=i
            oldj=j
##            #if eventCount is 0:
##                eventCount += 1
##                print('\n')
##                logging.info('Event Detected ')
##                logging.info('Chunk #: "%s", Starts at: "%s"'% (i, t))
##                logging.info('rms: "%s", MULTIPLIER * longtermRMS: "%s"\n' % (rms, abs(MULTIPLIER*longtermAvgRms)))
##                ax[1].plot(t, rms, marker='.', color='r')
                
        #logging.info('dB: "%s", rms: "%s", longtermAverage "%s", longtermAvgRMS: "%s"' % (decibel, rms, longtermAvgVol,longtermAvgRms))
        #logging.info('Peak Freq: "%s"' % peakFreq)

        i += int(seg_length)
        j += int(seg_length)
                
    plt.xlabel('Time')
    plt.ylabel('Volume - RMS')
    plt.title('Event Detection, Window Size: "%s", Events Detected: "%s"' % (WINDOW_SIZE, eventCount))
    plt.legend(loc='best')  
    plt.savefig('volume.png', bbox_inches='tight')
    logging.info('eventCount: "%s"' % eventCount)
    print('\n')

def rmsAnalysisOverlapping():
    '''
    Basic testing for volume event detection - Overlapping windows by 50%, events temporally linked
    '''
    MULTIPLIER = 1.75
    WINDOW_SIZE = .5
    eventCount=0
    
    #First file is for RMS refernece, Second file is for event analyszis
    backFrames, backSamp = getRecordedFrames('C:\\Users\\...')
    frames, sampFreq = getRecordedFrames('C:\\Users\\...')

    #Information on the audio file to analyze
    lenData = float(len(frames))                                    #Samples
    howManyChunks = math.floor((lenData/sampFreq)/WINDOW_SIZE)      #(lenData/sampFreq)=Duration of recording. (Duration/WindowSize)=# of windows
    whereToBreak = int(lenData/howManyChunks)                       #Size of Windows in samples

    #RMS threshold
    longtermAvgVol = getDb(backFrames)
    longtermAvgRms = getRms(backFrames)

    #Plot the horizontal line for RMS threshold and the signal in time
    f,ax = plt.subplots(2,figsize=(12,10), dpi=100, sharex=True)
    timeArray = np.linspace(0, (lenData/ sampFreq), lenData)
    ax[0].plot(timeArray, frames)
    ax[0].set_title('Audio Signal')
    ax[0].set_ylabel('Amplitude')
    timeArray = np.linspace(0, (lenData/ sampFreq))
    horizLine = np.array([MULTIPLIER*longtermAvgRms for i in range(len(timeArray))])   
    plt.plot(timeArray, horizLine, label='Detection Level: '+str(MULTIPLIER)+'x')

    #Window overlap - /2 == 50%
    seg_length = whereToBreak
    i, j = 0, seg_length
    step = seg_length / 2

    #To connect events that are temporally adjacent 
    oldi=0
    oldj=0

    frames_chunk=[]
    
    while j <= (howManyChunks*whereToBreak):
        framesChunk = frames[i: j]
        t = ((i/sampFreq) + (j/sampFreq)) / 2

        rms = getRms(framesChunk)
        plt.plot(t, rms, marker='.', color='k')

        if rms>(MULTIPLIER*longtermAvgRms):
            #logging.info('i: "%s", oldj: "%s"' % (i, oldj))
            if i<oldj:
                #events are the same if within a window size of each other - connect with red line and draw red dot
                plt.plot(t, rms, marker='o', color='r')
                a=[[t, rms],[oldx,oldy]]
                plt.plot(*zip(*a), marker='o', linestyle='-',color='r')
            else:
                #Draw new red point for event
                eventCount += 1
                print('\n')
                logging.info('Event Detected ')
                logging.info('i time: "%s", j time: "%s"' % ((i/sampFreq),(j/sampFreq)))
                logging.info('rms: "%s", MULTIPLIER * longtermRMS: "%s"\n' % (rms, abs(MULTIPLIER*longtermAvgRms)))
                plt.plot(t, rms, marker='o', color='r')
                oldx=t
                oldy=rms
            oldi=i
            oldj=j
            
        i += int(step)
        j += int(step)

    plt.xlabel('Time')
    plt.ylabel('Volume - RMS')
    plt.title('Event Detection, Window Size: "%s", Events Detected: "%s"' % (WINDOW_SIZE, eventCount))
    plt.legend(loc='best')  
    plt.savefig('volumeWindowed.png', bbox_inches='tight')
    logging.info('eventCount: "%s"' % eventCount)



if __name__ == "__main__":
    rmsAnalysisNoOverlapping()
    rmsAnalysisOverlapping()

    ##Testing these for event detection
    #zcr = getZeroCrossing(frames)
    #logging.info('ZCR: %s' % zcr)

    #logEnergy = getLogarithmicEnergy(frames)
    #logging.info('LogEnergy: "%s"' % logEnergy)

    #result = getShortTimeFourierTransform(frames, sampFreq)
    #img = plt.imshow(result, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
    #plt.savefig('shortTimeFourier.png', bbox_inches='tight')
