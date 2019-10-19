#Written by Mitchell Bourke - bourkey08@gmail.com
#Renders a mandlebrot fractal using the cuda framework
#Requirements
#   Pillow
#   Numpy
#   Numba
#   Cuda framework installed from nvidia

from numba import cuda, jit
import numpy, time, os, numba
from PIL import Image

#We need to specify the path to the cuda runtime
os.environ['CUDA_HOME']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1'

#@jit(nopython=True)
@cuda.jit
def MandleBrot(size, cords, escape):
    key = 0
    iterations = 0xfffff

    pos = cuda.grid(1)

    y = pos % size[1]
    x = (pos - y) / size[1]

    
    #Convert x and y into points between the 2 specified cords
    x1 = numba.float32((((cords[1][0] - cords[0][0]) / size[0]) * x ) + cords[0][0])
    y1 = numba.float32((((cords[1][1] - cords[0][1]) / size[1]) * y ) + cords[0][1])

    #Create a complex number from x1 and y1
    z = complex(x1, y1)

    c = z

    #Calculate the time to escape for this point               
    n = 0

    for n in xrange(0, iterations):            
        if abs(z) > 2:
            break

        z = z*z + c

    escape[x * size[1] + y] = n

@jit(forceobj=True)
def SaveImage(imgsize, data):
    #Turn the computed time to escape values into an image
    img = Image.new('RGB', imgsize)
    imgp = img.load()

    for x in xrange(0, imgsize[0]):
        for y in xrange(0, imgsize[1]):
            n = data[x * imgsize[1] + y]
            
            red = n % 256
            green = (n >> 8) % 256
            blue = (n >> 16) % 256
            
            imgp[x, y] = (red, green, blue)
            
    img.save('Test.png', 'PNG')
    
if __name__ == '__main__':
    #Set Config Variables
    threadsperblock = 512
    #imgsize = [1920 * 16, 1080 * 16]
    imgsize = [2000, 2000]
    cordinates = [[-2.0, 1.3], [0.6, -1.3]]

    #Define the cordinates of the image as well as its size
    size = numpy.array(imgsize, numpy.int32)    
    cords = numpy.array(cordinates, numpy.float32)
    data = numpy.zeros(size[0] * size[1], numpy.int32)
    
    escape = cuda.to_device(data)
    
    #Calculate the number of thread blocks in the grid
    blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
    
    start = time.time()
    
    MandleBrot[blockspergrid, threadsperblock](size, cords, escape)
    
    print time.time() - start

    data = escape.copy_to_host()
    
    SaveImage(imgsize, data)

    
