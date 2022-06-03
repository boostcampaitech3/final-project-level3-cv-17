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
	# 추출된 이미지의 mask가 전부 00이 될 경우
	# 1.png의 경우 filter base로 추출하면 전부 0이 나옴.
	# 그럴 경우 오류 발생
	c2=np.min(x)
	c1=np.max(x)
	r2=np.min(y)
	r1=np.max(y)
	print((r1,c1,r2,c2))
	return (r1,c1,r2,c2)


def replace_sky(img, img_mask, ref): # , ref_mask
	height, width = img_mask.shape
	sky_resize = cv2.resize(ref, (width, height))
	
	mask_bool = (img_mask/255).astype(np.uint8)
	mask_bool_reverse = (1-img_mask/255).astype(np.uint8)
	new_img = np.zeros_like(img)
	new_img += np.repeat(mask_bool[:,:,np.newaxis], 3, axis=2) * sky_resize
	new_img += np.repeat(mask_bool_reverse[:,:,np.newaxis], 3, axis=2) * img

	return new_img

	# height, width = img_mask.shape
	# sky_resize = cv2.resize(ref, (width, height))

	# I_rep = img.copy()
	# sz = img.shape

	# for i in range(sz[0]):
	# 	for j in range(sz[1]):
	# 		if(img_mask[i,j].any()):
	# 			I_rep[i,j,:] = sky_resize[i,j,:]
	# return I_rep
# def replace_sky(img,img_mask,ref,ref_mask):

# 	# rect mask filtering
# 	y2,x2,y1,x1=find_min_sky_rect(ref_mask)
# 	print(y2,x2,y1,x1)
# 	# 이 min조건을 단순히 거는게 아니라 이걸 걸어서 0이 되거나
# 	# treshold를 못넘으면 제거되게 하면 되겠다.
# 	roi = ref[y1:y2, x1:x2]

# 	r1,c1,r2,c2=find_max_sky_rect(img_mask)
# 	height=r1-r2+1
# 	width=c1-c2+1

# 	sky_resize = cv2.resize(roi, (width, height))

# 	I_rep=img.copy()
# 	sz=img.shape

# 	for i in range(sz[0]):
# 			for j in range(sz[1]):
# 				if(img_mask[i,j].any()):
# 					I_rep[i,j,:] = sky_resize[i-r2,j-c2,:]
# 	return I_rep

def guideFilter(I, p, mask_edge, winSize, eps):	#input p,giude I
    
	I=I/255.0
	p=p/255.0
	mask_edge=mask_edge/255.0
	#I的均值平滑
	mean_I = cv2.blur(I, winSize)
    
    #p的均值平滑
	mean_p = cv2.blur(p, winSize)
    
    #I*I和I*p的均值平滑
	mean_II = cv2.blur(I*I, winSize)
    
	mean_Ip = cv2.blur(I*p, winSize)
    
    #方差
	var_I = mean_II - mean_I * mean_I#方差公式
    
    #协方差
	cov_Ip = mean_Ip - mean_I * mean_p
   
	a = cov_Ip / (var_I + eps)
	b = mean_p - a*mean_I
    
    #对a、b进行均值平滑
	mean_a = cv2.blur(a, winSize)
	mean_b = cv2.blur(b, winSize)

	q=p.copy()
	sz=mask_edge.shape
	# edge=mask_edge.copy()
	kernel=np.ones((5,5),np.uint8)
	# kernel=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],np.uint8)
	edge=cv2.dilate(mask_edge,kernel)

	# edge8=edge*255
	# edge8=edge8.astype(np.uint8)
	# mask_edge8=mask_edge*255
	# mask_edge8=mask_edge8.astype(np.uint8)
	# cv2.imwrite('d:/MyLearning/DIP/Final_Project/Report/mask_edge.png',mask_edge8)
	# cv2.imwrite('d:/MyLearning/DIP/Final_Project/Report/edge.png',edge8)

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
