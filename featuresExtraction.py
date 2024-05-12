import numpy as np
from scipy.signal import convolve2d

# Local phase quantization descriper
# this function gets the LPQ features from the image into a vector of size 255
def lpq(img):
    # the window size that we will slide the image with
    winSize=3
    # alpha in STFT approaches
    STFTalpha=1/winSize
    
    # Compute descriptor responses only on part that have full neigborhood
    convmode='valid' 
    
    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    w0=np.ones_like(x)
    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
    w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]
    LPQdesc=LPQdesc/LPQdesc.sum()
    
    return LPQdesc

# this function gets the features of array of images and returns
# features matrix, each row is an example i, each column is feature j
def getFeaturesList(images):
    lpqFeatures = []
    for i in range(len(images)):
        lpqFeatures.append(lpq(images[i]))

    return (np.asarray(lpqFeatures))