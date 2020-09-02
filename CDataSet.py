import cv2
import numpy as np
import xlsxwriter

def SkinColorUpper (Hue,mult1,mult2):
    upper = [Hue,mult1*255,mult2*255]
    upper = np.array(upper)
    return upper

def SkinColorLower (Hue,mult1,mult2):
    lower = [Hue,mult1*255,mult2*255]
    lower = np.array(lower)
    return lower

direc = ('Fotos/Prueba19.jpg')
img = cv2.imread(direc,1)
# cv2.imshow("Imagen",img)
# cv2.waitKey(0)
heigth, width = img.shape[:2]
#print (img.shape)

start_row,start_col = int(0),int(0)
end_row, end_col = int(heigth), int(width*.3)
img = img[start_row:end_row,start_col:end_col]

hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

mask = cv2.inRange(hls,SkinColorLower(0,0.2,0),SkinColorUpper(27,0.7,0.75))
#mask = cv2.inRange(hls,np.array([0,49,61]),np.array([20,255,127]))

blur = cv2.medianBlur(mask,5)

ret,edges = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
edges= cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
draw = cv2.drawContours(img,contours,-1,(255,0,0),3)

cv2.imshow("mask",draw)
cv2.waitKey(0)



