#Written by Mitchell Bourke - bourkey08@gmail.com
#Renders a nebulabrot fractal using the cuda framework
#Requirements
#   Pillow
#   Numpy
#   Numba
#   Cuda framework installed from nvidia

from numba import cuda, jit
import numpy, time, os, numba
from PIL import Image


#Need to specify the path to the cuda runtime
os.environ['CUDA_HOME']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1'

@cuda.jit(fastmath=True)
def MandleBrot(size, cords, iterations, escape):
    key = 0

    pos = cuda.grid(1)

    y = pos % size[1]
    x = (pos - y) / size[1]

    for xstep in xrange(2):
        for ystep in range(2):
            xstep = xstep / 2
            ystep = ystep / 2
            
            #Convert x and y into points between the 2 specified cords
            x1 = numba.float32((((cords[1][0] - cords[0][0]) / size[0]) * (x + xstep) ) + cords[0][0])
            y1 = numba.float32((((cords[1][1] - cords[0][1]) / size[1]) * (y + ystep) ) + cords[0][1])
            
            #Create a complex number from x1 and y1
            z = complex(x1, y1)

            #Set c to be the same as z
            c = z
            
            #First lets check if this point escapes in the given number of iterations
            escapes = False
            
            n = 0
            
            for n in range(iterations):            
                if abs(z) > 2:
                    escapes = True
                    break

                z = z*z + c

            #Now we know if the point escapes, if it does iterate over it again and increment each point that is crossed by 1
            if escapes:        
                z = c#Reset z to the value of the origonal complex number

                for n in range(iterations):            
                    if abs(z) > 2:#If is greater than 2 it has escaped so stop iterating
                        break

                    z = z*z + c

                    #Increment the current position
                    xpos = abs(int((z.real - cords[1][0]) / ((cords[0][0] - cords[1][0]) / size[0])))
                    ypos = abs(int((z.imag - cords[1][1]) / ((cords[0][1] - cords[1][1]) / size[1])))
                    arraypos = xpos * size[1] + ypos
                    
                    #Increment relavant position in the array, we need to check if the position is valid as z may be off the canvas
                    if arraypos < len(escape) and arraypos > 0:
                        escape[arraypos] += 1
            
@jit(forceobj=True)
def SaveImage(imagepixels, filename='Test'):
    #Turn the computed time to escape values into an image
    img = Image.new('RGB', (len(imagepixels), len(imagepixels[0])))
    imgp = img.load()

    for x in xrange(0, img.size[0]):
        for y in xrange(0, img.size[1]):           
            imgp[x, y] = (imagepixels[x, y, 0], imagepixels[x, y, 1], imagepixels[x, y, 2])
            
    img.save(filename + '.png', 'PNG')

#Takes the returned 1d array and an array of [r,g,b] and updates the specified channel
@jit
def UpdateArrayWithVal(data, array, pos):
    #Iterate over all x and y pixels, get the value that corresponds to that cell and apply it to the specific color channel
    for x in range(len(array)):
        for y in range(len(array[0])):
            n = data[x * len(array[0]) + y]
            array[x, y, pos] = n % 256
    
    
if __name__ == '__main__':
    #Set Config Variables    
    #This sets the number of threads in a block, threads in the same block can talk via shared memory and sync there status via syncthreads
    threadsperblock = 1024

    #Specify the size of the image as well as the cordinates on the complex plain that we should map the image pixels to
    imgsize = [4000, 4000]
    cordinates = [[-1.9, 1.2], [0.9, -1.2]]

    #Specify the number of iterations that will be used to render each channel of the image (Red, Green, Blue)
    itercounts = [1024, 0, 8192*4]
    #itercounts = [0xff, 0xffff, 0xfffff]   
    #itercounts = [8192, 0, 0]
    #itercounts = [0, 0, 8192]
    
    #Define an array that we will build the image pixels in, this will be a 3d array [x,y][rgb as 0-2]
    imagepixels = numpy.zeros((imgsize[0], imgsize[1], 3), numpy.int32)
    
    #Define arrays that will be copied to the gpu
    size = numpy.array(imgsize, dtype=numpy.int32)
    cords = numpy.array(cordinates, dtype=numpy.float32)

    #Calculate the number of thread blocks in the grid
    blockspergrid = (size[0] * size[1] + (threadsperblock - 1)) // threadsperblock

    start = time.time()

    for i in xrange(3):
        #Define a data array on the gpu
        
        escape = cuda.device_array(imgsize[0] * imgsize[1], dtype=numpy.int32)       
        
        MandleBrot[blockspergrid, threadsperblock](size, cords, itercounts[i], escape)
        
        #Explicitly copy the data array back from the gpu
        data = escape.copy_to_host()
        
        UpdateArrayWithVal(data, imagepixels, i)
    
    print time.time() - start

    #Save The image to disk with the specified filename
    SaveImage(imagepixels)  
