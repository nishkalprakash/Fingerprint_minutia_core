# %%
import numpy as np;
import cv2;
import numpy.matlib
from scipy import ndimage;
from scipy import signal
from BI import Datasets,DS_PREFIX, Parallel

def ridge_orient(im, gradientsigma, blocksigma, orientsmoothsigma):
    rows,cols = im.shape;
    #Calculate image gradients.
    sze = np.fix(6*gradientsigma);
    if np.remainder(sze,2) == 0:
        sze = sze+1;
        
    gauss = cv2.getGaussianKernel(int(sze),gradientsigma);
    f = gauss * gauss.T;
    
    fy,fx = np.gradient(f);     #Gradient of Gaussian
    
    #Gx = ndimage.convolve(np.double(im),fx);
    #Gy = ndimage.convolve(np.double(im),fy);
    
    Gx = signal.convolve2d(im,fx,mode='same');    
    Gy = signal.convolve2d(im,fy,mode='same');
    
    Gxx = np.power(Gx,2);
    Gyy = np.power(Gy,2);
    Gxy = Gx*Gy;
    
    #Now smooth the covariance data to perform a weighted summation of the data.    
    
    sze = np.fix(6*blocksigma);
    
    gauss = cv2.getGaussianKernel(int(sze),blocksigma);
    f = gauss * gauss.T;
    
    Gxx = ndimage.convolve(Gxx,f);
    Gyy = ndimage.convolve(Gyy,f);
    Gxy = 2*ndimage.convolve(Gxy,f);
    
    # Analytic solution of principal direction
    denom = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps;
    
    sin2theta = Gxy/denom;            # Sine and cosine of doubled angles
    cos2theta = (Gxx-Gyy)/denom;
    
    
    if orientsmoothsigma:
        sze = np.fix(6*orientsmoothsigma);
        if np.remainder(sze,2) == 0:
            sze = sze+1;    
        gauss = cv2.getGaussianKernel(int(sze),orientsmoothsigma);
        f = gauss * gauss.T;
        cos2theta = ndimage.convolve(cos2theta,f); # Smoothed sine and cosine of
        sin2theta = ndimage.convolve(sin2theta,f); # doubled angles
    
    orientim = np.pi/2 + np.arctan2(sin2theta,cos2theta)/2;
    return(orientim);

# %%
def normalise(img,mean,std):
    normed = (img - np.mean(img))/(np.std(img));    
    return(normed)
    
def ridge_segment(im,blksze,thresh):
    
    rows,cols = im.shape;    
    
    im = normalise(im,0,1);    # normalise to get zero mean and unit standard deviation
    
    
    new_rows =  int(blksze * np.ceil((float(rows))/(float(blksze))))
    new_cols =  int(blksze * np.ceil((float(cols))/(float(blksze))))
    
    padded_img = np.zeros((new_rows,new_cols));
    stddevim = np.zeros((new_rows,new_cols));
    
    padded_img[0:rows][:,0:cols] = im;
    
    for i in range(0,new_rows,blksze):
        for j in range(0,new_cols,blksze):
            block = padded_img[i:i+blksze][:,j:j+blksze];
            
            stddevim[i:i+blksze][:,j:j+blksze] = np.std(block)*np.ones(block.shape)
    
    stddevim = stddevim[0:rows][:,0:cols]
                    
    mask = stddevim > thresh;
    
    mean_val = np.mean(im[mask]);
    
    std_val = np.std(im[mask]);
    
    normim = (im - mean_val)/(std_val);
    
    return(normim,mask)

# %%

def walkonce(im, mask, dfim, start, step, Td):
    sp = []
    current =0
    path = np.array([start])
    while True:
        temp = path[current,:]
        ori = dfim[int(temp[0]), int(temp[1])]
        current = current + 1
        path = np.concatenate([path, temp + step*(np.around([[-np.sin(ori), np.cos(ori)]]))], axis=0)
        if (any(path[current,:] < 1) or any((path[current,:] - im.shape) >= 0) or not mask[int(path[current,0]),int(path[current,1])]):
            break
        cpath = path[0:current,:]
        dcpath = cpath - np.matmul(np.ones((cpath.shape[0],1)), np.array([path[current,:]]))
        sqart = np.sqrt(dcpath[:,0]**2 + dcpath[:,1]**2)
        sqart = np.transpose(sqart).reshape((1,-1))
        sqart = np.fliplr(sqart)
        indx = np.argwhere(sqart < Td)
        if min(indx.shape)==0:
            ind = np.empty(shape=(0,0))
        else:
            ind = sqart.shape[1] - indx[0][1]
        if ind.size !=0:
            if current == ind:
                sp = path[current,:]
            elif (current - ind) <=11:
                sp =  np.around(np.sum(path[ind:current+1,:], axis=0)/(current - ind + 1))
            break
    return sp

def checkstable(im, mask, orientim, tempsp, step, Td, R):
    stable=0
    tempsp = np.array([tempsp])
    
    if min(tempsp.shape) !=0:
        stable=1
        trystart = np.matmul(np.ones((4,1)), tempsp) + np.array([[0,-1], [-1,0], [0,1], [1,0]])*R
        for j in range(0,4):
            if (any(trystart[j,:] < 1) or any((trystart[j,:] - im.shape) >= 0) or not mask[int(trystart[j,0]),int(trystart[j,1])]):
                stable=0
                break
            newsp = walkonce(im, mask, orientim, trystart[j,:], step, Td)
            newsp = np.array([newsp])
            if min(newsp.shape) == 0 or np.linalg.norm(tempsp - newsp) > R:
                stable=0
                break
    return stable


def mergeneighbors(points, threshold):

    for i in range(0,points.shape[0]-1):
        if points[i,0] == 0:
            continue
        pointi = points[i,:]
        for j in range(i+1,points.shape[0]):
            if points[j,0] == 0:
                continue
            if (np.linalg.norm(points[i,:] - points[j,:]) < threshold):
                pointi = np.concatenate([pointi, points[j,:]], axis=0)
                points[j,:] = [0, 0]
        if len(pointi)>=2:
            pointi = pointi.reshape((-1,2))
        s = np.sum(pointi, axis=0)
        points[i,:] = np.around(np.array([[s[0], s[1]]])/pointi.shape[0])
    points = points[~np.all(points==0, axis=1)]
    return points


def walking(img_path):
    img = cv2.imread((Path(DS_PREFIX)/img_path).as_posix(),0)
    #initializing the dictionary
    doc={'path':img_path}
    sps = {}
    sps['core'] = []
    sps['delta'] = []
    step = 7
    n = 2
    Td = 2
    R = 16

    blksze = 16;
    thresh = 0.3;
    normim, mask = ridge_segment(img,blksze,thresh);    
    mask[(mask.shape[0]//blksze)*blksze+1:mask.shape[0],:] = 0
    mask[:,(mask.shape[1]//blksze)*blksze+1:mask.shape[1]] = 0


    orientim = ridge_orient(normim, 1, 3, 3)
    orientim = np.pi - orientim
    
    #sampling starting point

    I,J = np.where(mask==1)
    edge0 = np.array([min(I),min(J)])
    edge1 = np.array([max(I),max(J)])
    d = (edge1-edge0)//(n+1)
    sampled_rows = np.array(range(edge0[0]+d[0], edge0[0] + d[0]*n + 1,d[0]))    #If some error comes, consider +1 also
    sampled_cols = np.array(range(edge0[1]+d[1], edge0[1] + d[1]*n + 1,d[1]))
    sampled_points = np.transpose([[np.kron(sampled_rows, np.ones((1,n)))],[np.matlib.repmat(sampled_cols, 1, n)]])
    sampled_points = np.reshape(sampled_points,(4,2))
    


    #Detect Cores
    for r in range(0,4):
        if len(sps['core']) != 0:
            break
        WDFc1 = 2.0*orientim + (r+1)*np.pi/2
        core1 = np.array([[]])
        core2 = np.array([[]])
        for i in range(0,sampled_points.shape[0]):
            p = sampled_points[i,:]
            if (not mask[int(p[0]),int(p[1])]):
                continue
            if min(core1.shape) == 0:
                tempsp = walkonce(img, mask, WDFc1, p, step, Td)
                if checkstable(img, mask, WDFc1, tempsp, step, Td, R):
                    core1 = tempsp
            if min(core2.shape) == 0:
                tempsp = walkonce(img, mask, WDFc1-np.pi, p, step, Td)
                if checkstable(img, mask, WDFc1-np.pi, tempsp, step, Td, R):
                    core2 = tempsp
        if min(core1.shape)==0:
            sps['core'] = np.array([core2])
        elif min(core2.shape)==0:
            sps['core'] = np.array([core1])
        else:
            sps['core'] = np.concatenate([core1,core2], axis=0)
            


    #Detect Deltas
    for r in range(0,4):
        if len(sps['delta']) != 0:
            break
        WDFd = -2.0*orientim + (r+1)*np.pi/2
        for i in range(0,sampled_points.shape[0]):
            p = sampled_points[i,:]
            if (not mask[int(p[0]),int(p[1])]):
                continue
            tempsp = walkonce(img, mask, WDFd, p, step, Td)
            if checkstable(img, mask, WDFd, tempsp, step, Td, R):
                sps['delta'] = np.concatenate([sps['delta'],tempsp], axis=0)
        if len(sps['delta']) >=2:
            sps['delta'] = sps['delta'].reshape((-1,2))

    sps['core'] = np.array([sps['core']])
    sps['delta'] = np.array([sps['delta']])
    
    sps['core'] = sps['core'].reshape((-1,2))
    sps['delta'] = sps['delta'].reshape((-1,2))

    sps['core'] = np.fliplr(mergeneighbors(sps['core'],20)).tolist()
    sps['delta'] = np.fliplr(mergeneighbors(sps['delta'],20)).tolist()
    doc['sp_walking'] = sps
    return doc

# %%
# Pass an image and get core points
from pathlib import Path
# base = 

if __name__ == "__main__":
        
    ds=Datasets("BI")
    pl = ["FVC2000","FVC2002","FVC2004","FVC2006"]
    for p in pl:
        p=Path(p)
        con=ds.connect(p.as_posix())
    
        # images = list(base.glob("*.tiff"))
        images = [doc["path"] for doc in con.find({"sp_walking":{"$exists":False}},{"path":1,"_id":0})]
        # thres = 25
        ll = Parallel(debug=False)
        # to_process = ((i,thres) for i in images)
        # for i in images:
        for docs in ll(walking,images,batch_size=1,chunksize=1):
            # push to mongodb con
            # mv=feature_vector(i, thres)
            # con.insert_one(doc)
            for doc in docs:
                path = doc.pop('path')
                con.update_one({'path':path},{"$set":doc},upsert=False)