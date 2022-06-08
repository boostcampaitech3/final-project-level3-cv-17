# -*- coding: utf-8 -*-
"""
Created on Sun May 30 17:28:21 2021
@author: OwYeong
"""
import	cv2
import	numpy	as	np
from	matplotlib	import	pyplot	as	pt
from pykuwahara import kuwahara
import os
import glob
import argparse
import time
import imutils



#Function Define Start
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def get_dark_channel(image_color, window_size):
    """
    Dark channel prior
    Select the mininum value of r,g,b channel within a sliding window.
    """
    b,g,r = cv2.split(image_color)# Retrive the blue, green and red channel
    
    minimum_bgr_channel = cv2.min(cv2.min(b,g),r);# get minimum value between red,green, and blue channel, This basically generate a Grayscale image
        
    # NOTE: Erode in grayscale image, will help to select the lowest intensity value in the sliding window
    structure_element = cv2.getStructuringElement(cv2.MORPH_RECT,(window_size,window_size))
    dark_channel = cv2.erode(minimum_bgr_channel,structure_element) 
    
    return dark_channel#return result

def is_sky_exist_in_image(imgcolor):
    # Section: Generating dark channel
    darkChannel = get_dark_channel(imgcolor,105)# use a large window size to reduce the effect of artificial light
    thresholdOtsu,thresholdedDarkChannel = cv2.threshold(darkChannel.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Make the otsu threshold more bias, as we want to get more on sky area.
    adjustedThreshold = int(thresholdOtsu * 0.65)
    thresholdOtsu,thresholdedDarkChannel = cv2.threshold(darkChannel.astype(np.uint8),adjustedThreshold,255,cv2.THRESH_BINARY)
    

    num_of_labels, labeled_img = cv2.connectedComponents(thresholdedDarkChannel)
    M, N = labeled_img.shape
    
    is_sky_exist = False
    threshold = int(N * (7/10)) # 7/10 of the total column must stick to top border to indicate a sky exist
    
    # Section: Check if sky exist
    # If 
    # at least 1 labelled region stick to the (threshold) of total image top border OR
    # at least 1 labelled region has 80% of the image width AND
    # the labelled does not contain any pixel that stick to the bottom border of the image.
    # Then Sky exist.
    for label in range(1, num_of_labels):
        # for each labelled segment in image
        label_in_current_region = np.zeros((M, N), dtype=np.uint8)
        label_in_current_region[labeled_img == label] = 1 # generate the binary image of current labelled segment
        
        labelledRegionWidthSum = np.sum(label_in_current_region, axis=1) 
        number_of_pixel_stick_at_top_border = labelledRegionWidthSum[0] # In Current labelled segment, check number of pixel that stick to the top border of image
        max_width_of_labelled_region = np.max(labelledRegionWidthSum)
        
        isNoPixelTouchesBorderBottom = (labelledRegionWidthSum[-1] == 0)
        
        if (number_of_pixel_stick_at_top_border > threshold or max_width_of_labelled_region > int(N*0.8)) and isNoPixelTouchesBorderBottom:
            # current labelled segment, meet threshold
            is_sky_exist = True # sky exist

            return is_sky_exist # return result
           
    #After iterating all label, none of the segment meet the threshold. 
    return is_sky_exist #return result
        
        
def variance_filter(img_gray, window_size = 3): # osth treshold랑 비슷한 느낌인데
    """
    Variance filter
    Calculate the variance in the sliding window
    """
    img_gray = img_gray.astype(np.float64)
    
    # Variance calculate using vectorized operation. As using a 2d loop to slide the window is too time consuming.
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (window_size, window_size), borderType=cv2.BORDER_REPLICATE) for x in (img_gray, img_gray*img_gray))
    return wsqrmean - wmean*wmean

def find_hsv_upper_lower_threshold_for_skyarea_and_weather_condition(rough_sky_mask, hls_img):
     """
     This function will estimate the upper and lower threshold of sky are in hue, lightness, saturation channel 
     based on the rough_sky_mask provided. It also estimate the weather condition of the image using hls image 
     """
     rough_sky_area_hls = cv2.bitwise_and(hls_img, hls_img, mask=rough_sky_mask) # extracted rough sky area
    # mask 영역에서 서로 겹치는 부분을 추출한다. 애초에 저 마스크 영역으로 제한되어있고
    # 거기에 대해서 hls_img가 겹치는 부분을 추출한다.
    # 즉 그냥 더 넓은 영역이라고 생각해도 좋을 것 같다.
     M, N, _ = hls_img.shape
     h, l, s = cv2.split(rough_sky_area_hls)
     
     hlist = np.full((M, N), -1, dtype=np.float64)
     llist = np.full((M, N), -1, dtype=np.float64)
     slist = np.full((M, N), -1, dtype=np.float64)
    
     # Section: Calculate statistical information for hue, lightness, saturation channel
     hlist[rough_sky_mask == 255] = h[rough_sky_mask == 255] # copy pixel in answer mask to the list
     llist[rough_sky_mask == 255] = l[rough_sky_mask == 255] # copy pixel in answer mask to the list
     slist[rough_sky_mask == 255] = s[rough_sky_mask == 255] # copy pixel in answer mask to the list
    
     value_tolerance = 50 # tolerance value for light
     
     hflatten = hlist.flatten()[hlist.flatten() >= 0 ] #  extract the pixels in answer mask and flat the array 
     hflatten.sort()
     
     sflatten = slist.flatten()[slist.flatten() >= 0 ] #  extract the pixels in answer mask and flat the array 
     sflatten.sort()
     
     s_removed_noise = sflatten[int(sflatten.size*0.05): int(sflatten.size - int(sflatten.size*0.05))] # trim 5 percent from forward and backward direction
     
     saturation_max = 0 if (sflatten.size == 0) else s_removed_noise.max()
     saturation_min = 0 if (sflatten.size == 0) else s_removed_noise.min()   
     
     lflatten = llist.flatten()[llist.flatten() >= 0 ]  #  extract the pixels in answer mask and flat the array 
     lflatten.sort()
     
     l_removed_noise = lflatten[int(lflatten.size*0.05): int(lflatten.size - int(lflatten.size*0.05))] # trim 5 percent from forward and backward direction
     
     lightness_mean = 0 if (lflatten.size == 0) else int(np.mean(l_removed_noise))
     lightness_max = 0 if (lflatten.size == 0) else lflatten.max()
     lightness_min = 0 if (lflatten.size == 0) else lflatten.min()
     
     # Section: Estimate lightness threshold for sky area
     if lightness_mean> 200:
         #Day images, which high in light as the mean value of lightness in sky area is extremely high
         weather_condition = "day"
         lUpper = int(lightness_mean + value_tolerance)
         lLower = int(150) # sky is extremely bright, hence the minimum lightness would be 150 based on experiment
     elif lightness_mean> 100:
         weather_condition = "day"
         # Day image, but lower in light
         # Hence, we use max and min to 
         lUpper = int(lightness_max + value_tolerance)
         lLower = int(lightness_min - value_tolerance)
     else:
         weather_condition = "night"
         #Night images, as mean value of lightness is less than 100
         lUpper = int(lightness_max + value_tolerance)
         lLower = int(lightness_min - 10)
          
     # Generate hls upper and lower threshold based on lightness channel as we found that lightness is most representative for a sky area
     hls_lower_threshold = np.array([0,8 if lLower < 8 else lLower,0])
     hsv_upper_threshold = np.array([255,255 if lUpper > 200 else lUpper,255])
     
     # If saturation is high, it is most likely cloudy as clouds is not saturated most of the time while sky area is saturated
     if np.abs(saturation_max - saturation_min) > 120 and weather_condition == "day" and lightness_mean<240:
         weather_condition = "dayCloudy"
         
     if np.abs(saturation_max - saturation_min) > 50 and weather_condition == "night":
         weather_condition = "nightCloudy"
     
     return hls_lower_threshold, hsv_upper_threshold, weather_condition # 이런 정보들로부터 날씨 정보를 뽑는다.
     # 추가적으로 hls 정보를 뽑는다. 
     # 
     
     

def generate_final_sky_mask(initialSkyMask):
    num_of_labels, labeled_img = cv2.connectedComponents(initialSkyMask)
    
    M, N = labeled_img.shape
    
    largest_labelled_region = None
    
    # Section: Find Largest labelled region in initialSkyMask
    for label in range(1, num_of_labels):
        # for each labelled segment in image
        label_in_current_region = np.zeros((M, N), dtype=np.uint8)
        label_in_current_region[labeled_img == label] = 1 # generate the binary image of current labelled segment
        
        number_of_pixel_in_current_region = np.sum(label_in_current_region)
        if largest_labelled_region is None:
            largest_labelled_region = label_in_current_region
        else:
            if number_of_pixel_in_current_region > np.sum(largest_labelled_region):
                largest_labelled_region = label_in_current_region   
    
    # Section: Noise filtering. If black pixels is surrounded by white pixel, change to white pixel
    indicies_of_all_labelled_region = np.argwhere(largest_labelled_region == 1)
    
    # Generate padding before performing
    # Sky area in initialSkyMask will have a padding of 1 while non-sky area will have a padding of 0
    max_row_in_labelled_region_left_border = None
    max_row_in_labelled_region_right_border = None
    
    if len(indicies_of_all_labelled_region[indicies_of_all_labelled_region[:,1]==0, :][:,0]) > 0:
        max_row_in_labelled_region_left_border = np.max(indicies_of_all_labelled_region[indicies_of_all_labelled_region[:,1]==0, :][:,0])
    if len(indicies_of_all_labelled_region[indicies_of_all_labelled_region[:,1]==N-1, :][:,0]) > 0:
        max_row_in_labelled_region_right_border = np.max(indicies_of_all_labelled_region[indicies_of_all_labelled_region[:,1]==N-1, :][:,0])
    
    padding_size = 1
    paddedlargest_labelled_region = np.pad(largest_labelled_region, ((padding_size,padding_size),(padding_size,padding_size)), 'constant')
    paddedlargest_labelled_region[0, :] = 1 # top border is 1
    
    if max_row_in_labelled_region_left_border is not None:
        paddedlargest_labelled_region[0:max_row_in_labelled_region_left_border, 0] = 1 # left border padding is 1, up to the max row in labelled region
    if max_row_in_labelled_region_right_border is not None:
        paddedlargest_labelled_region[0:max_row_in_labelled_region_right_border, N+1] = 1 # right border padding is 1, up to the max row in labelled region

   
    complement_of_largest_region = 1 - largest_labelled_region #invery
    num_of_labels_in_largest_labelled_region, complement_of_largest_region_labelled = cv2.connectedComponents(complement_of_largest_region)
    padded_complement_of_largest_region_labelled = np.pad(complement_of_largest_region_labelled, ((padding_size,padding_size),(padding_size,padding_size)), 'constant') # allow spaces for dilation for pixel that stick to border
        
    for label in range(1,num_of_labels_in_largest_labelled_region):
        # for each black colored segment in image, we check wheter is surrounded by white pixel
        label_in_current_region = np.zeros((M, N), dtype=np.uint8)
        label_in_current_region[complement_of_largest_region_labelled==label] = 1
        paddedlabel_in_current_region = np.pad(label_in_current_region, ((padding_size,padding_size),(padding_size,padding_size)), 'constant') # allow spaces for dilation for pixel that stick to border
        
        crossSe = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # 3x3 cross kernel
        dilatedlabel_in_current_region = cv2.dilate(paddedlabel_in_current_region, crossSe,iterations = 1)
        
        surrounded_pixel = dilatedlabel_in_current_region - paddedlabel_in_current_region # find surrounded pixel by using dilation

        
        if np.min(paddedlargest_labelled_region[surrounded_pixel == 1]) == 1:
            # the current segment is surrounded by regionOfInterest
            paddedlargest_labelled_region[padded_complement_of_largest_region_labelled==label] = 1 # make the black segment as region of interst as it is surrounded by Roi
    
    return largest_labelled_region, paddedlargest_labelled_region[padding_size:M+1, padding_size:N+1 ] # exlude padding area
    
def find_sky_mask(img_color_bgr):
    # Step 1: Check whether sky exist.
    # if is_sky_exist_in_image(img_color_bgr) == False:
    #     if DEBUG_MODE:
    #         print("NO Sky Exist returning empty binary mask")
    #     # Sky is not exist, return a empty binary mask to indicate there is no sky
    #     return np.zeros((img_color_bgr.shape[0], img_color_bgr.shape[1]), dtype=np.uint8)
    
    # Step 2: Filter image with kuwahara filter to reduce noise while preserving edges
    # img_bgr_kuwahara_filtered = kuwahara(img_color_bgr, method='mean', radius=1)
    img_bgr_kuwahara_filtered = img_color_bgr
    # Step 3: Resize image to a smaller resolution, for faster estimation of hsv upper and lower threshold, and the weather condition
    resized_img_bgr_kuwahara_filtered = resize_with_aspect_ratio(img_bgr_kuwahara_filtered,width=300) # Resize the image, to reduce processing time
    
    originalM, originalN, _ = img_color_bgr.shape
    resizedM, resizedN, _ = resized_img_bgr_kuwahara_filtered.shape
    
    # Step 4: Estimate the hls upper and lower threshold, and the weather condition 
    resized_img_hls_kuwahara_filtered = cv2.cvtColor(resized_img_bgr_kuwahara_filtered, cv2.COLOR_BGR2HLS)
    
    # we perform dark channel prior with a large window to reduce the effect of artificial light at night
    resized_dark_channel = get_dark_channel(resized_img_bgr_kuwahara_filtered,105) 
    
    # OTSU threshold is performed on dark channel prior. 
    # As non-sky area have at least one channel(r,g,b) is dark, After OTSU, the binary mask is a rough sky area
    thresholdSelected,rough_sky_area_mask = cv2.threshold(resized_dark_channel.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Estimate the hsv upper and lower threshold, and the weather condition, with the rough sky mask
    hls_lowerthreshold, hls_upperthreshold, weather_condition = find_hsv_upper_lower_threshold_for_skyarea_and_weather_condition(rough_sky_area_mask,resized_img_hls_kuwahara_filtered)
    
    
    img_hls_kuwahara_filtered = cv2.cvtColor(img_bgr_kuwahara_filtered, cv2.COLOR_BGR2HLS)
    img_rgb_kuwahara_filtered = cv2.cvtColor(img_bgr_kuwahara_filtered, cv2.COLOR_BGR2RGB) 
    img_gray_kuwahara_filtered = cv2.cvtColor(img_bgr_kuwahara_filtered, cv2.COLOR_BGR2GRAY)
    
    r,g,b = cv2.split(img_rgb_kuwahara_filtered)   
    h,l,s = cv2.split(img_hls_kuwahara_filtered)    
    
    # Step 5: Generate the hls-based sky mask, based on estimated threshold 
    hlsBasedSkyMask = np.zeros((originalM,originalN), dtype=np.uint8)
    hlsBasedSkyMask = cv2.inRange(img_hls_kuwahara_filtered, hls_lowerthreshold, hls_upperthreshold)
    
    if weather_condition == "day" or weather_condition == "dayCloudy":
        # For Both day and cloudy image, we need to include a white mask, as very often day images sky or cloud is pure white
        whiteMask = cv2.inRange(img_hls_kuwahara_filtered, np.array([0,220,0]), np.array([255,255,200]))
        hlsBasedSkyMask = cv2.bitwise_or(hlsBasedSkyMask,whiteMask) # combine
    
    
    # Step 6: Generate the variance-based sky mask
    varianceFilteredOrgImg = variance_filter(img_gray_kuwahara_filtered, window_size=3) # variance filter with sliding window
    
    varianceThreshold = 10 # if the variance in the image fall belows or equal to the threshold, then the pixel is considered as sky pixel
    
    # Based on experiment, 
    # We found that, 
    # 10 work best for day as in day image, sky area color may changes from blue to white which would higher the variance. Fortunately, in day, non-sky region are mostly non homogeneous and have a variance more than 10.
    # 5 work best for night as in night images, non-sky area is also having a low variance due to insufficient light and sky is having a more lower variance. 
    # 150 work best for cloudy day, as cloud will increase the variance in sky. We higher the threshold to reduce the impact of cloud
    # 20 work best for cloudy night,similarly, We higher the threshold to reduce the impact of cloud
    if weather_condition == "day":
        varianceThreshold = 10
    if weather_condition == "dayCloudy":
        varianceThreshold = 150 # threshold is increased for cloudy images, as a low threshold will make generate outline by the cloud
    if weather_condition == "night":
        varianceThreshold = 5 # threshold is decreased, as the variance in night image for sky is low. This could avoid missing sky pixel
    if weather_condition == "nightCloudy":
        varianceThreshold = 20 # a low threshold will make generate outline by the cloud
    
    
    varianceBasedSkyMask = np.zeros((originalM,originalN), dtype=np.uint8)
    varianceBasedSkyMask[varianceFilteredOrgImg <= varianceThreshold] = 1
    
    # Step 7: Generate a initial sky mask, by logical and with varianceBasedSkyMask and hlsBasedSkyMask
    skyMask = cv2.bitwise_and(varianceBasedSkyMask,hlsBasedSkyMask)
    
    # Step 8: Fine tune the initial sky mask by excluding the confident non-sky area
    # Based on image analysis of all skyfinder dataset, we found that
    # blue channel less than 9, green channel less than 4, and light channel less than 8 are guaranteed to be non-sky area.
    not_sky_mask = np.zeros((originalM,originalN), dtype=np.uint8)
    not_sky_mask[(b <9) | (g <4) | (l<=8)] = 1 #confident not sky
    
    sky_mask_after_futher_elimination = skyMask.copy()
    sky_mask_after_futher_elimination[ not_sky_mask == 1] = 0
    
    
    squareSe = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # 3x3 RECT kernel
    
    # Step 9: Perform opening to remove small noise in fine_tuned_sky_mask
    skymask_opening = cv2.morphologyEx(sky_mask_after_futher_elimination,cv2.MORPH_OPEN,squareSe, iterations=3)
    
    # Step 10: Perform closing on non-sky area to bridge small gap between non-sky area in fine_tuned_sky_mask
    # we are performing closing on non-sky area, hence we invert the sky mask
    skymask_closing_non_roi = cv2.morphologyEx(1 - skymask_opening,cv2.MORPH_CLOSE,squareSe, iterations=3) 
    
    # After closing, invert the inverted mask.
    initial_sky_mask = 1 - skymask_closing_non_roi
        
    # Step 11: Generate final mask from the initial sky mask. 
    # Basically, choose the largest connected area in the mask, and remove all black pixel surrounded by white pixel to reduce the edges of cloud
    largest_labelled_region, final_sky_mask = generate_final_sky_mask(initial_sky_mask)
    
    if DEBUG_MODE:
        """
        Display Output by each step
        """
        pt.figure()
        pt.subplot(3,4,1).set_title("Original image")
        pt.imshow( cv2.cvtColor(img_color_bgr, cv2.COLOR_BGR2RGB))
        pt.subplot(3,4,2).set_title("Kuwahara filtered image")
        pt.imshow(cv2.cvtColor(resized_img_bgr_kuwahara_filtered, cv2.COLOR_BGR2RGB))
        pt.subplot(3,4,3).set_title("Dark channel Prior")
        pt.imshow(resized_dark_channel, cmap="gray")
        pt.show()
        pt.subplot(3,4,4).set_title("OTSU's Dark channel Prior \nfor hls threshold and weather estimation")
        pt.imshow(rough_sky_area_mask,cmap="gray")
        pt.show()
        pt.subplot(3,4,5).set_title("HLS-based sky mask")
        pt.imshow(hlsBasedSkyMask,cmap="gray")
        pt.show()
        pt.subplot(3,4,6).set_title("Variance-based sky mask")
        pt.imshow(varianceBasedSkyMask,cmap="gray")
        pt.show()
        pt.subplot(3,4,7).set_title("Combined HLS-based & Variance-based sky mask")
        pt.imshow(skyMask,cmap="gray")
        pt.show()
        pt.subplot(3,4,8).set_title("Fine-tuned the initial sky mask \nby excluding the confident non-sky area")
        pt.imshow(skymask_opening,cmap="gray")
        pt.show()
        pt.subplot(3,4,9).set_title("Opening on sky area in initial sky mask")
        pt.imshow(skymask_opening,cmap="gray")
        pt.show()
        pt.subplot(3,4,10).set_title("Closing on non-sky area in initial sky mask")
        pt.imshow(initial_sky_mask,cmap="gray")
        pt.show()
        pt.subplot(3,4,11).set_title("Select Largest area in initial sky mask")
        pt.imshow(largest_labelled_region,cmap="gray")
        pt.show()
        pt.subplot(3,4,12).set_title("Final sky mask after removing all black pixel\nsurrounded by white pixel in largest area")
        pt.imshow(final_sky_mask, cmap="gray")
        pt.show()
        pt.waitforbuttonpress()
        pt.close("all")#Close the all matlib plot
    
    return final_sky_mask * 255 # convert (0 and 1) to (0 and 255)

def process_image_or_folder(args_received):
    if isinstance(args_received, argparse.Namespace):
        image_path = args_received.image_path
        
    # Find the input image provided
    if os.path.isfile(image_path):
        # The input is a single image
        paths = [image_path]
        output_directory = './' # output at the current directory of current script
        
    
    if len(paths) == 0:
        #throw exception
        raise Exception("No .jpg images found in the directory provided: " + image_path)
        
    for image_path in paths:
        start_time = time.time() # record start time for current image
        current_image_name = os.path.basename(image_path)
        current_image = cv2.imread(image_path,1)

        height, width = current_image.shape[:2]
        if height >= width:
            current_image = imutils.resize(current_image,height=1280) 
        else:
            current_image = imutils.resize(current_image,width=1280) 

        M, N, _ = current_image.shape
        sky_mask = find_sky_mask(current_image)
        
        output_path = os.path.join(output_directory, "{}_sample_output.jpg".format(os.path.splitext(current_image_name)[0]))
        # cv2.imwrite(output_path, sky_mask)
        
        print("Processed ", image_path)
        print("Image dimension(width x height): {}x{}".format(N, M))
        print("Segmentation Filter base Processing time: {} seconds".format(time.time() - start_time))
    
    return sky_mask
"""
**Note: Only run on IDE when DEBUG_MODE is True
"""
DEBUG_MODE = False
IMAGE_PATH = "9483/20130611_135417.jpg" # image path must be manually set, when DEBUG_MODE is true
# Use '9483/*.jpg' for whole folder

if __name__ == '__main__':
    if not DEBUG_MODE:
        # DEBUG MODE IS OFF
        parser = argparse.ArgumentParser(
            description='Parser for sky segmentation script')
    
        parser.add_argument('--image_path', type=str,
                            help='path to a test image or folder of images', required=True)
        
        argument_received =  parser.parse_args()
        start_time = time.time()
        
        num_of_images_processed = process_image_or_folder(argument_received)
        print("\n\nTotal Image Processed: {} image".format(num_of_images_processed))
        print("Total Processing time: {} seconds".format(time.time() - start_time))
        print("Average Processing time: {} image".format((time.time() - start_time)/num_of_images_processed))
    else:
        # DEBUG MODE IS ON
        print(
        "DEBUG MODE IS ON:",
        "- All preview plot or log will be shown in this mode.",
        "- It is recommend to process a single images in DEBUG_MODE to understand more about the sky segmentation process",
        "- Please avoid processing a whole folder using debug mode as every image process will generate\n",
        "- Image Processing time in Debug mode is not accurate due to pause duration on matplotlib.\n",
        sep="\n")
        
        start_time = time.time()
        
        num_of_images_processed = process_image_or_folder(IMAGE_PATH)
        print("\n\nTotal Image Processed: {} image".format(num_of_images_processed))
        print("Total Processing time: {} seconds".format(time.time() - start_time))
        print("Average Processing time: {} image".format((time.time() - start_time)/num_of_images_processed))