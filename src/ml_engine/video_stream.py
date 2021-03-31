from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import os
import cv2
import pytesseract
import re
import distance
import time
from tqdm import tqdm


args = {}
args["east"] = "models/frozen_east_text_detection.pb"
args["min_confidence"] = 0.5 # Default = 0.5
args["padding_y"] = 0.1
args["padding_x"] = 0.8
args["language"] = 'eng'
args["oem"] = 1 # OCR Engine Mode
args["psm"] = 3 # Page Segmentation Mode

scale = 36


args["margin"] = 5
args["debug"] = False


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def valid_text(text):
	text = clean_text(text)
	# print(text)
	if text == "":
		return False, ''
	p = re.compile('((\d{3})-(\d{3})-(\d{2}))')
	if p.search(text) is not None:
		found = p.search(text).group()
		print(f"Matched regex 1: {found}")
		return True, found
	p = re.compile('(LPN\d{8})')
	if p.match(text) is not None:
		# print(f"Matched regex 2: {p.match(text).group()}")
		return True, p.match(text).group()
	else:
		if len(text) > 3:
			min_distance_levenshtein = 2
			index = text.find('L')
			if index >= 0:		
				#print('trazas ',distance.levenshtein('LPN',text[max(index-2,0):index]))
				if distance.levenshtein('LPN',text[index:min(index+2,len(text))]) <= min_distance_levenshtein:
					text_aux = text[index+3:-1]
					text_aux = 'LPN'+text_aux
					#print("NUEVO LPN L ", text_aux)
					if p.match(text_aux) is not None:
						# print(f"Matched regex 2: {p.match(text_aux).group()}")
						return True, p.match(text_aux).group()

			index = text.find('P')
			if index >= 0:		
				#print('trazas ',distance.levenshtein('LPN',text[max(index-2,0):index]))
				if distance.levenshtein('LPN',text[max(index-1,0):min(index+1,len(text))]) <= min_distance_levenshtein:
					text_aux = text[index+2:-1]
					text_aux = 'LPN'+text_aux
					#print("NUEVO LPN P ", text_aux)
					if p.match(text_aux) is not None:
						# print(f"Matched regex 2: {p.match(text_aux).group()}")
						return True, p.match(text_aux).group()

			index = text.find('N')
			if index >= 0:		
				#print('trazas ',distance.levenshtein('LPN',text[max(index-2,0):index]))
				if distance.levenshtein('LPN',text[max(index-2,0):index]) <= min_distance_levenshtein:
					text_aux = text[index+1:-1]
					text_aux = 'LPN'+text_aux
					#print("NUEVO LPN N ", text_aux)
					if p.match(text_aux) is not None:
						# print(f"Matched regex 2: {p.match(text_aux).group()}")
						return True, p.match(text_aux).group()
	return False, ''

def clean_text(text):
	text = text.upper()
	text = text.replace(" ", "")
	text = text.replace("U", "0")
	text = text.replace("O", "0")
	text = text.replace("S", "5")
	text = text.replace("Z", "2")
	text = text.replace("B", "8")
	text = text.replace("E", "P")
	text = text.replace("F", "P")
	import unicodedata
	text = unicodedata.normalize('NFKD', text)
	return text

def list_roi(roi, config, startX, startY, endX, endY, results, debug=False):
	# USE ROI Basic
	
	roi_basic = roi ## roi_basic = process_roi(roi)

	# text = pytesseract.image_to_string(roi_basic, config=config)
	text = pytesseract.image_to_string(roi_basic, config=config)
	# print("ROI text:", text)

	# add the bounding box coordinates and OCR'd text to the list
	# of results
	results.append(((startX, startY, endX, endY), text))
	roi_basic = cv2.putText(roi_basic, text, (startX, startY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
	if debug:
		cv2.imshow("ROI",roi_basic)

	key = cv2.waitKey(1) & 0xFF

	# USE ROI GAUSSIAN
	roi_gaussian = process_roi_gaussian(roi)

	# text = pytesseract.image_to_string(roi_gaussian, config=config)
	text = pytesseract.image_to_string(roi_gaussian, config=config)

	# add the bounding box coordinates and OCR'd text to the list
	# of results
	results.append(((startX, startY, endX, endY), text))
	roi_gaussian = cv2.putText(roi_gaussian, text, (startX, startY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
	if debug:				
		cv2.imshow("ROI",roi_gaussian)

	key = cv2.waitKey(1) & 0xFF

	return results

def process_roi(roi):
	# Transform ROI
	gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
	_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	#if np.mean(binary) > 127:
	#	binary = cv2.bitwise_not(binary)
  
	# Rotate image
	#rows,cols = binary.shape
	#M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
	#binary = cv2.warpAffine(binary,M,(cols,rows))
	#https://github.com/aparande/OCR-Preprocessing/blob/master/tutorial.ipynb
	kernel = np.ones((1, 1), np.uint8)
	
	#binary = cv2.erode(binary, kernel, iterations=1)
	binary = cv2.dilate(binary, kernel, iterations=1)

	#binary = cv2.erode(binary, kernel, iterations=1)
	#binary = cv2.dilate(binary, kernel, iterations=1)
	#binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
	#binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
	return binary

def process_roi_gaussian(roi):
	# Transform ROI
	gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
	
	# use adaptive thresh gaussian
	binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	
	#if np.mean(binary) > 127:
	#	binary = cv2.bitwise_not(binary)
  
	# Rotate image
	#rows,cols = binary.shape
	#M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
	#binary = cv2.warpAffine(binary,M,(cols,rows))
	#https://github.com/aparande/OCR-Preprocessing/blob/master/tutorial.ipynb
	kernel = np.ones((1, 1), np.uint8)
	
	binary = cv2.erode(binary, kernel, iterations=2)
	binary = cv2.dilate(binary, kernel, iterations=2)

	#binary = cv2.erode(binary, kernel, iterations=1)
	#binary = cv2.dilate(binary, kernel, iterations=1)
	#binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
	#binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)	
	return binary

def package_path(*paths, package_directory=os.path.dirname(os.path.abspath(__file__))):
    return os.path.join(package_directory, *paths)

def process_video(video_path: str, debug=False):
	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...", args["east"])
	net = cv2.dnn.readNet(package_path(args["east"]))
	print("[INFO] loaded EAST text detector...")

	vs = cv2.VideoCapture(video_path)
	total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	if debug:
		out = cv2.VideoWriter(f'output_{time.time()}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 8, (1280*2, 720))

	# loop over frames from the video stream
	print("Starting loop")
	counter = 0
	with tqdm(total=total_frames) as pbar:
		while True:
			# grab the current frame, then handle if we are using a
			# VideoStream or VideoCapture object
			image = vs.read()
			counter+=1
			pbar.update(1)
			# print(counter, "/", total_frames)
			image = image[1] #if args.get("video", False) else image
			# check to see if we have reached the end of the stream
			if image is None:
				print("Video end")
				break
			if counter % 5 != 0:
				continue

			orig = image.copy()
			display_orig = orig.copy()
			(origH, origW) = image.shape[:2]
		
			# set the new width and height and then determine the ratio in change
			# for both the width and height
			# Important: The EAST text requires that your input image dimensions be multiples of 32
			# so if you choose to adjust your --width  and --height  values, ensure they are multiples of 32!
			aspect_ratio = origW/origH
			(newW, newH) = (32*scale, 32* int(scale/aspect_ratio))
			rW = origW / float(newW)
			rH = origH / float(newH)

			image = cv2.resize(image, (newW, newH))
			(H, W) = image.shape[:2]
			print()
			# Convert image to grayscale
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = np.zeros_like(image)
			image[:,:,0] = gray
			image[:,:,1] = gray
			image[:,:,2] = gray
			
			# Increase contrast
			#-----Converting image to LAB Color model----------------------------------- 
			lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

			#-----Splitting the LAB image to different channels-------------------------
			l, a, b = cv2.split(lab)

			#-----Applying CLAHE to L-channel-------------------------------------------
			clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
			cl = clahe.apply(l)

			#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
			limg = cv2.merge((cl,a,b))

			#-----Converting image from LAB Color model to RGB model--------------------
			image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

			display_image = image.copy()
			
			# construct a blob from the image and then perform a forward pass of
			# the model to obtain the two output layer sets
			blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
										(123.68, 116.78, 103.94), swapRB=True, crop=False)
			net.setInput(blob)
			(scores, geometry) = net.forward(layerNames)

			# decode the predictions, then  apply non-maxima suppression to
			# suppress weak, overlapping bounding boxes
			(rects, confidences) = decode_predictions(scores, geometry)
			boxes = non_max_suppression(np.array(rects), probs=confidences)

			# initialize the list of results
			results = []
			# loop over the bounding boxes
			padded_boxes = []
			for (startX, startY, endX, endY) in boxes:
				if debug:
					cv2.rectangle(display_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
				# scale the bounding box coordinates based on the respective
				# ratios
				startX = int(startX * rW)
				startY = int(startY * rH)
				endX = int(endX * rW)
				endY = int(endY * rH)

				# in order to obtain a better OCR of the text we can potentially
				# apply a bit of padding surrounding the bounding box -- here we
				# are computing the deltas in both the x and y directions
				dX = int((endX - startX) * args["padding_x"])
				dY = int((endY - startY) * args["padding_y"])

				# apply padding to each side of the bounding box, respectively
				startX = max(0, startX - dX - args["margin"])
				startY = max(0, startY - dY - args["margin"])
				endX = min(origW, endX + (dX * 2) + args["margin"])
				endY = min(origH, endY + (dY * 2) + args["margin"])

				padded_boxes.append((startX, startY, endX, endY))

				# if debug:
				# 	cv2.rectangle(display_orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
				
			boxes = non_max_suppression(np.array(padded_boxes))
			for (startX, startY, endX, endY) in boxes:
				if debug:
					cv2.rectangle(display_orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
				# extract the actual padded ROI
				roi = orig[startY:endY, startX:endX]

				# in order to apply Tesseract v4 to OCR text we must supply
				# (1) a language, (2) an OEM flag of 4, indicating that the we
				# wish to use the LSTM neural net model for OCR, and finally
				# (3) an OEM value, in this case, 7 which implies that we are
				# treating the ROI as a single line of text
				lang = args["language"]
				oem = args["oem"]
				psm = args["psm"]
				config = (f"-l {lang} --oem {oem} --psm {psm} -c tessedit_char_whitelist=uUoOsSzZbBeEfFlLpPnN0123456789-")
				actual_results = list_roi(roi, config, startX, startY, endX, endY, results, debug)
				results = results + actual_results
				
			# sort the results bounding box coordinates from top to bottom
			results = sorted(results, key=lambda r: r[0][1])
			# print(results)
			
			# loop over the results
			for ((startX, startY, endX, endY), text) in results:
				# strip out non-ASCII text so we can draw the text on the image
				# using OpenCV, then draw the text and a bounding box surrounding
				# the text region of the input image
				# text: str = text
				# print(text.strip())
				text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
				# TODO:retornar texto limpio
				isValid ,result = valid_text(text)
				if not isValid:
					continue
				print("Found:",result)
				yield result
				if debug:
					cv2.rectangle(display_orig, (startX, startY), (endX, endY),
								(0, 255, 0), 2)
					cv2.putText(display_orig, result, (startX, startY - 20),
								cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
				
		
			display_orig = imutils.resize(display_orig, width=1280)
			display_image = imutils.resize(display_image, width=1280)
			h1, w1 = display_orig.shape[:2]
			h2, w2 = display_image.shape[:2]


			# #create empty matrix
			# vis = np.zeros((h1+h2, max(w1, w2),3), np.uint8)

			# #combine 2 images
			# vis[:h2, :w2,:3] = display_image
			# vis[h2:h2+h1, :w1,:3] = display_orig

			#create empty matrix
			vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

			#combine 2 images
			vis[:h1, :w1,:3] = display_orig
			vis[:h2, w1:w1+w2,:3] = display_image

			if debug:
				# cv2.imshow("Text Detection", orig)
				# cv2.imshow("Image",display_image)
				cv2.imshow("Orig",vis)
				out.write(vis)
		
		
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
	if debug:		
		out.release()
	print("END")
