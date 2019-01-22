from skimage import io 
import numpy as np

def Bresenham(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D > 0:
            y += 1
            D -= dx
        D += dy

def GetDetectorsCoordinates(radius, alfa, fi, detectorQty):
    xs = np.zeros(detectorQty, dtype=np.int)
    ys = np.zeros(detectorQty, dtype=np.int)
    for i in range(0, detectorQty):
        x = radius * np.cos(alfa + np.pi - fi/2 + i*fi/(detectorQty-1))
        y = radius * np.sin(alfa + np.pi - fi/2 + i*fi/(detectorQty-1))
        xs[i] = int(round(x, 0))
        ys[i] = int(round(y, 0))
    return (xs, ys)
    
def GetEmiterCoordinate(radius, alfa,offset=0):
    x = radius * np.cos(alfa)+offset
    y = radius * np.sin(alfa)+offset
    return (int(round(x, 0)), int(round(y, 0)))
    
def GetSinogramPixel(emiter, detector, img):
    hitPixels = list(Bresenham(emiter[0], emiter[1], detector[0], detector[1]))
    result = 0.0    
    
    for i in hitPixels:
        #print((int(len(img)/2)-1,int(len(img)/2)-1))  
        #print(str(i[0]) + " : " + str(len(img)/2))
        result += img[i[0] + int(len(img)/2)-1, i[1] + int(len(img)/2)-1]
    return result / len(hitPixels)

def GetSinogramLine(radius, alfa, fi, detectorQty,img):
    emiter=GetEmiterCoordinate(radius, alfa)
    detectors=GetDetectorsCoordinates(radius, alfa, fi, detectorQty)
    line=np.array([])
    
    for detector in zip(detectors[0],detectors[1]):
        line=np.append(line,GetSinogramPixel(emiter,detector,img))
    return line

def GetSinogram(steps,radius, fi, detectorQty,img):
    sinogram=np.empty((0,detectorQty))
    n=2*np.pi/steps
    for i in range(steps):
        line=GetSinogramLine(radius, n*i, fi, detectorQty,img)
        sinogram=np.vstack((sinogram,line))
    
    return sinogram

def FillLine(scan,value,emiter,detector):
    
    points=list(Bresenham(emiter[0], emiter[1], detector[0],detector[1]))
    
    for point in points:
        scan[point[0],point[1]]+=value
    
    return scan

def FillScan(steps,radius, fi,sinogram):
    
    n=2*np.pi/steps
    detectorsQty=len(sinogram[0])
    
    
    scan=np.zeros((2*radius,2*radius))
    
    for row,line in enumerate(sinogram):
        emiter=GetEmiterCoordinate(radius, row*n,radius-1)

        detectors=GetDetectorsCoordinates(radius, row*n,fi , detectorsQty)
        for detector in zip(zip(detectors[0]+radius-1,detectors[1]+radius-1),line):
            FillLine(scan,detector[1],emiter,detector[0])
    
    return scan
def normalizeMatrix(mat):
    norm=mat.max()
 
    for i,k in enumerate(mat):

        for j,f in enumerate(mat[0]):

            mat[i,j]=mat[i,j]/norm
def convFunc(position,rng):
    if position==0:
        return 1
    if position%2==0:
        return 0
    
    return (-4/np.pi**2)/position**2
    
def convolutionArray(rng):
    arr=np.array([])
    for i in range(2*rng):
        arr=np.append(arr,convFunc(i-rng,rng))
    return arr

def tomoFilter(sinogram):
    
    convFunc=convolutionArray(int(len(sinogram[0])/2))
    sinEd=np.empty((0,len(sinogram[0])))
    for i in sinogram:
        line=np.convolve(i,convFunc,'same')
        sinEd=np.vstack((sinEd,line))
    return sinEd

img = io.imread("img/Kwadraty2.jpg", as_grey=True)
rad=int(min(len(img),len(img[0]))/2)
det=180
emit=180

sinogram=GetSinogram(steps=emit,radius=rad, fi=np.pi/4, detectorQty=det,img=img)

scan=FillScan(emit,rad,np.pi/2,tomoFilter(sinogram))
normalizeMatrix(scan)
io.imshow(scan)

