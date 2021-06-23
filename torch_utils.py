from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import glob
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import argparse
import imutils
import time



async def get_classification(image):
    # Detect table
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 11))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(grad, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    tableCnt = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(tableCnt)
    table = image[y:y + h, x:x + w]

    # Resize table image
    scale_percent = 90
    width = int(table.shape[1] * scale_percent / 100)
    height = int(table.shape[0] * scale_percent / 100)
    table = cv2.resize(table, (width, height), interpolation = cv2.INTER_AREA)

    # Read text and save to csv
    # Dataframe to store result
    n_lines = 26
    result = pd.DataFrame(
        columns = ['Date','摘要(お客様メモ)','お支払全額', 'お預り全額', '差引残高', '記号・店番号'],
        index = [*range(n_lines)]
    )
    # Cut each column and save to seperate image
    height, weight , _ = table.shape
    columns_px = [0,59, 59, 145, 151, 255, 259, 375, 375, 480, 480, 535]
    columns_px = [int(i / 0.5 * scale_percent / 100) for i in columns_px]

    ####### RUN COGNITIVE OCR ##########
    subscription_key = "d461a1c90bba449982b9ec4c1e792498"
    endpoint = "https://ocrforwealth.cognitiveservices.azure.com/"

    images_folder = 'images/'
    for j in range(result.shape[1]):
        crop_img = table[int(68 / 0.5 * scale_percent / 100):int(595/0.5 * scale_percent / 100), columns_px[j*2]:columns_px[j*2+1]]
        cv2.imwrite(images_folder + 'crop_img{}.png'.format(j), crop_img)

        # initiate top and bottom pixel
        height, weight , _ = crop_img.shape
        cel_height = int(height/n_lines)
        start = 0

        computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

        read_image_path = images_folder + 'crop_img{}.png'.format(j)
        read_image = open(read_image_path, "rb")
        read_response = computervision_client.read_in_stream(read_image,language='ja', raw=True)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for the retrieval of the results
        while True:
          read_result = computervision_client.get_read_result(operation_id)
          if read_result.status.lower () not in ['notstarted', 'running']:
              break
          print ('Waiting for result...')
          time.sleep(1.5)

        # Print results to csv
        j = int(read_image_path.split('/')[-1].split('.')[0][-1])
        if j == 0:
            text = [line.text.replace(' ','-') for line in read_result.analyze_result.read_results[0].lines]
        elif j==2:
            text = [line.text.replace('+','*') for line in read_result.analyze_result.read_results[0].lines]

        elif j == 4:
            text = [''.join(c for c in line.text.replace('|','') if c.isdigit()) for line in read_result.analyze_result.read_results[0].lines]
        else:
            text = [line.text.replace('|','') for line in read_result.analyze_result.read_results[0].lines]
        b_box = [line.bounding_box for line in read_result.analyze_result.read_results[0].lines]

        # Loop by line
        for i in range(26):
            top = start + cel_height*i
            bottom = start + cel_height*(i+1)
            for idx, box in enumerate(b_box):
                if ((box[1] in range(top, bottom+1))|(box[7] in range(top, bottom+1))):
                    result.iloc[i, j] = text[idx]
                    text.pop(idx)
                    b_box.pop(idx)
                else:
                    result.iloc[i, j] = ''
                break

    result.to_csv(images_folder+'result.csv', index = False)

    return result
