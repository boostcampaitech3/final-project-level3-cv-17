import numpy as np
import cv2

def find_min_sky_rect(sky_mask):
    x_max = sky_mask.shape[1]
    for index,i in enumerate(sky_mask):
        if sky_mask.shape[1] > len(np.where(i!=0)[0]):
            break
    y_max = index
    return y_max,x_max,0,0


def find_max_sky_rect(mask):
	

	index=np.where(mask!=0)
	index= np.array(index,dtype=int)
	y=index[0,:]
	x=index[1,:]

	c2=np.min(x)
	c1=np.max(x)
	r2=np.min(y)
	r1=np.max(y)
	print((r1,c1,r2,c2))
	return (r1,c1,r2,c2)
def replace_sky(img,img_mask,ref,ref_mask):

	height, width = img_mask.shape

	sky_resize = cv2.resize(ref, (width, height))

	I_rep=img.copy()
	sz=img.shape

	for i in range(sz[0]):
		for j in range(sz[1]):
			if(img_mask[i,j].any()):
				I_rep[i,j,:] = sky_resize[i,j,:]
	return I_rep

def guideFilter(I, p, mask_edge, winSize, eps):	#input p,giude I
    
	I=I/255.0
	p=p/255.0
	mask_edge=mask_edge/255.0

	mean_I = cv2.blur(I, winSize)
    
	mean_p = cv2.blur(p, winSize)
    
	mean_II = cv2.blur(I*I, winSize)
    
	mean_Ip = cv2.blur(I*p, winSize)
    
	var_I = mean_II - mean_I * mean_I
    
	cov_Ip = mean_Ip - mean_I * mean_p
   
	a = cov_Ip / (var_I + eps)
	b = mean_p - a*mean_I
    
	mean_a = cv2.blur(a, winSize)
	mean_b = cv2.blur(b, winSize)

	q=p.copy()
	kernel=np.ones((5,5),np.uint8)
	edge=cv2.dilate(mask_edge,kernel)

	q[edge==1]=mean_a[edge==1]*I[edge==1]+mean_b[edge==1]
			
	q = q*255
	q[q>255] = 255
	q = np.round(q)
	q = q.astype(np.uint8)
	return q

def color_transfer(source, mask_bw, target, mode):

	source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype("float32")

	(l, a, b) = cv2.split(target)
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	if mode:
		index=np.where(mask_bw==0)
		index= np.array(index,dtype=int)
		sz=index.shape

		fl = target[:,:,0][mask_bw==0]
		fa = target[:,:,1][mask_bw==0]
		fb = target[:,:,2][mask_bw==0]
		
		(lMeanTar_1, lStdTar_1) = (np.mean(fl), np.std(fl,ddof=1))
		(aMeanTar_1, aStdTar_1) = (np.mean(fa), np.std(fa,ddof=1))
		(bMeanTar_1, bStdTar_1) = (np.mean(fb), np.std(fb,ddof=1))
		(lMeanTar_2, lStdTar_2, aMeanTar_2, aStdTar_2, bMeanTar_2, bStdTar_2) = image_stats(source)
		# (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
		
		l[mask_bw==0] -= lMeanTar_1
		a[mask_bw==0] -= aMeanTar_1
		b[mask_bw==0] -= bMeanTar_1

		alpha=0.5
		beta=1-alpha

		l[mask_bw==0] *= (alpha*lStdSrc+ beta*lStdTar_1) / lStdTar_1
		a[mask_bw==0] *= (alpha*aStdSrc+ beta*aStdTar_1) / aStdTar_1
		b[mask_bw==0] *= (alpha*bStdSrc+ beta*bStdTar_1) / bStdTar_1

		l[mask_bw==0] += alpha*lMeanSrc+beta*lMeanTar_1
		a[mask_bw==0] += alpha*aMeanSrc+beta*aMeanTar_1
		b[mask_bw==0] += alpha*bMeanSrc+beta*bMeanTar_1

	else:
		(lMeanTar_2, lStdTar_2, aMeanTar_2, aStdTar_2, bMeanTar_2, bStdTar_2) = image_stats(target)
		l -= lMeanTar_2
		a -= aMeanTar_2
		b -= bMeanTar_2

		l = (lStdSrc / lStdTar_2) * l
		a = (aStdSrc / aStdTar_2) * a
		b = (bStdSrc / bStdTar_2) * b
		l += lMeanSrc
		a += aMeanSrc
		b += bMeanSrc

	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
	
	# return the color transferred image
	return transfer

def image_stats(image):
	"""
	Parameters:
	-------
	image: NumPy array
		OpenCV image in L*a*b* color space

	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)

def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return: Dice index 2*TP/(2*TP+FP+FN)=2TP/(pred_P+true_P)
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))


import numpy as np
import scipy as sp
import scipy.ndimage


def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)


    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst

def _gf_color(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:,:,0], r) / N
    mI_g = box(I[:,:,1], r) / N
    mI_b = box(I[:,:,2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:,:,0]*p, r) / N
    mIp_g = box(I[:,:,1]*p, r) / N
    mIp_b = box(I[:,:,2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:,:,0] * I[:,:,0], r) / N - mI_r * mI_r;
    var_I_rg = box(I[:,:,0] * I[:,:,1], r) / N - mI_r * mI_g;
    var_I_rb = box(I[:,:,0] * I[:,:,2], r) / N - mI_r * mI_b;

    var_I_gg = box(I[:,:,1] * I[:,:,1], r) / N - mI_g * mI_g;
    var_I_gb = box(I[:,:,1] * I[:,:,2], r) / N - mI_g * mI_b;

    var_I_bb = box(I[:,:,2] * I[:,:,2], r) / N - mI_b * mI_b;

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
            ])
            covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
            a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = box(a, r) / N[...,np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1/s, order=1)
        Psub = sp.ndimage.zoom(p, 1/s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p


    (rows, cols) = Isub.shape

    N = box(np.ones([rows, cols]), r)

    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP


    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """ run a guided filter per-channel on filtering input p
        I - guide image (1 or 3 channel)
        p - filter input (n channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if p.ndim == 2:
        p3 = p[:,:,np.newaxis]

    out = np.zeros_like(p3)
    for ch in range(p3.shape[2]):
        out[:,:,ch] = _gf_colorgray(I, p3[:,:,ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out


def test_gf():
    import imageio
    cat = imageio.imread('cat.bmp').astype(np.float32) / 255
    tulips = imageio.imread('tulips.bmp').astype(np.float32) / 255

    r = 8
    eps = 0.05

    cat_smoothed = guided_filter(cat, cat, r, eps)
    cat_smoothed_s4 = guided_filter(cat, cat, r, eps, s=4)

    imageio.imwrite('cat_smoothed.png', cat_smoothed)
    imageio.imwrite('cat_smoothed_s4.png', cat_smoothed_s4)

    tulips_smoothed4s = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed4s[:,:,i] = guided_filter(tulips, tulips[:,:,i], r, eps, s=4)
    imageio.imwrite('tulips_smoothed4s.png', tulips_smoothed4s)

    tulips_smoothed = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed[:,:,i] = guided_filter(tulips, tulips[:,:,i], r, eps)
    imageio.imwrite('tulips_smoothed.png', tulips_smoothed)
