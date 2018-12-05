import numpy as np
import math
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.color import rgb2gray


def read_image(filename, representation):
    image = imread(filename)
    if(len(image.shape)<3):
        #the third dimension which indicates the colour-channels
        #is missing, meaning its a gray-scale image

        #convert it to float64
        im_float = image.astype(np.float64)
        im_float /= 255
        return im_float
    if(len(image.shape) == 3):
        #RGB image

        if(representation ==1):
            #convert to gray-scale
            im_g = rgb2gray(image)
            im_g = im_g.astype(np.float64)


            return im_g
        if representation ==2:
            im_f =  image.astype(np.float64)
            im_f /= 255

            return im_f


def dftFunc(size):
    x,y = np.meshgrid(np.arange(size),np.arange(size))
    omega = np.exp(-2*math.pi*1J/size)
    vanDerMonde = np.power(omega,x*y)
    return vanDerMonde

def DFT(signal):
    size = signal.shape[0]
    vanDerMonde = dftFunc(size)
    return np.dot(vanDerMonde,signal)

def IDFT(fourier_signal):
    size = fourier_signal.shape[0]
    inverseMatrix = np.linalg.inv(dftFunc(size))
    return (1/size)*(np.dot(inverseMatrix,fourier_signal)).astype(np.complex128)

def DFT2(image):
    return np.transpose(DFT(np.transpose(DFT(image).astype(np.complex128))))



def IDFT2(fourier_image):
    return np.transpose(IDFT(np.transpose(IDFT(fourier_image))))


def conv_der(im):
    conVector = [[-1,0,1]]
    xDerivative = convolve2d(im,conVector, 'same')
    yDerivative = convolve2d(im,np.transpose(conVector), 'same')
    magnitude = np.sqrt(np.abs(xDerivative)**2 + np.abs(yDerivative)**2)
    return magnitude


def fourier_der(im):
    sizeX = im.shape[0]
    sizeY = im.shape[1]

    xDerivative = DFT2(im)*(2*math.pi*1j)/sizeX
    yDerivative = DFT2(im)*(2*math.pi*1j)/sizeY

    #multiply each fourier coefficcient by u , while u is shifted to the center

    xDerivative = np.fft.fftshift(xDerivative)
    for i in range(sizeX):
        xDerivative[i] = xDerivative[i]*(-sizeX/2 +i)

    xDerivative = np.real(IDFT2(xDerivative))

    yDerivative = np.transpose(yDerivative)
    yDerivative = np.fft.fftshift(yDerivative)
    for i in range(sizeY):
        yDerivative[i] = yDerivative[i]*(-sizeY/2 +i)

    yDerivative = np.transpose(yDerivative)
    yDerivative = np.real(IDFT2(yDerivative))

    magnitude = np.sqrt(np.abs(xDerivative)**2 + np.abs(yDerivative)**2)
    return magnitude.astype(np.float64)


def blur_spatial(im,kernel_size):

    kernel2d = gaussKernel(kernel_size)
    return convolve2d(im, kernel2d,"same")





def blur_fourier(im,kernel_size):
    if kernel_size == 1:
        return im
    dftImage = DFT2(im)


    gaussiKer = gaussKernel(kernel_size)
    xCenter = math.floor(im.shape[0]/2 - math.floor(kernel_size/2))
    yCenter = math.floor(im.shape[1]/2 - math.floor(kernel_size/2))

    #check if the number of pixels within each image' dimension is odd or even and pss accordingly-->

    if im.shape[0]%2 == 0 :
        xPadding = (xCenter,xCenter-1)

    else:
        xPadding = (xCenter, xCenter)

    if im.shape[1]%2 == 0 :
        yPadding = (yCenter, yCenter -1)


    else:
        yPadding = (yCenter,yCenter)

    gaussiKer = np.pad(gaussiKer,(xPadding,yPadding),"constant")

    #transforming the padded gauss kernel into frequency domain and then multiply with image
    gaussKerDFT = DFT2(gaussiKer)
    filteredImageDFT =dftImage*gaussKerDFT
    #piece- wise multiply



    return np.fft.ifftshift(np.real(IDFT2(filteredImageDFT)))


def gaussKernel(kernel_size):
    GaussKernel = np.array([1,1])
    while len(GaussKernel) < kernel_size:
        GaussKernel  = np.convolve(GaussKernel,[1,1]).astype(np.float64) #using binomial coefficients


    kernel2d = convolve2d(GaussKernel.reshape(-1,1),np.transpose(GaussKernel).reshape(1,-1)).astype(np.float64)

    kernel2d = kernel2d/np.sum(kernel2d)#normalize the kernel
    return kernel2d.astype(np.float64)

