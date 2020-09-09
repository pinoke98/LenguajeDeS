import cv2
import numpy as np
import xlsxwriter
from glob import glob
import matplotlib.pyplot as plt

i=1

row=0
col=0

xlabel = []
ylabel = []
color = ["ro","bo"]

#Nuevo libro excel
databook = xlsxwriter.Workbook('DatosTrain.xlsx')
worksheet = databook.add_worksheet('Train')#nueva hoja

def SkinColorUpper (Hue,mult1,mult2):
    upper = [Hue,mult1*255,mult2*255]
    upper = np.array(upper)
    return upper

def SkinColorLower (Hue,mult1,mult2):
    lower = [Hue,mult1*255,mult2*255]
    lower = np.array(lower)
    return lower

for j in range(1,3):
    img_Carpeta = ('Fotos/'+str(j)+'/'+str(j)+' (*.jpg')
    img_names = glob(img_Carpeta)
    #print(img_names)
    for fn in img_names:
        print(fn)
        img = cv2.imread(fn,1)
        
        # cv2.imshow("Imagen",img)
        # cv2.waitKey(0)
        
        heigth, width = img.shape[:2]
        
        start_row,start_col = int(0),int(0)
        end_row, end_col = int(heigth), int(width*.3)
        img = img[start_row:end_row,start_col:end_col]
        
        hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)        
        mask = cv2.inRange(hls,SkinColorLower(0,0.2,0),SkinColorUpper(27,0.72,0.75))
        blur = cv2.medianBlur(mask,5)
        
        ret,edges = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        edges= cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        draw = cv2.drawContours(img,contours,-1,(255,0,0),3)
        
        cv2.imshow("Imagen",img)
        cv2.waitKey(0)
        
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        p = cv2.arcLength(c,True)
        empty = np.zeros((h,w),np.uint8)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi = edges[y:y+h,x:x+w]
        empty[0:h,0:w]=roi
        edges=empty
        
        cv2.imshow("Edges",edges)
        cv2.waitKey(0)
        
        img_resize = cv2.resize(edges,(100,100),interpolation=cv2.INTER_AREA)
        ret, edges_res = cv2.threshold(img_resize,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # cv2.imshow("img_resize",edges_res)
        # cv2.waitKey(0)
        
        contours_2,hierarchy_2 = cv2.findContours(edges_res.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #draw = cv2.drawContours(edges_res,contours,-1,(255,0,0),3)
        
        # cv2.imshow("img_resize",edges_res)
        # cv2.waitKey(0)
        
        for component_2 in zip(contours_2,hierarchy_2):
            currentContour =  component_2[0]
            x,y,w,h = cv2.boundingRect(currentContour)
            M = cv2.moments(currentContour)
            #print(M)
            cx = M['m10']/M['m00']
            cy = M['m01']/M['m00']
            A = cv2.contourArea(currentContour)
            p = cv2.arcLength(currentContour,True)
            aP=A/float(p*p)
            # print(M['m10'],M['m01'],M['m00'],cx,cy,A,p,aP)
            print(A)
            Hu = cv2.HuMoments(M)
            # print(Hu)
            VectorCarac = np.array([cx,cy,A,p,aP,Hu[0],Hu[1],Hu[2],Hu[3],Hu[4],Hu[5],Hu[6],M['m00'],M['m01'],M['m10'],M['m11']])
            for carac in (VectorCarac):
                    worksheet.write(row,col,str(j))
                    worksheet.write(row,i, carac)
                    i=i+1
            i=1
            row+=1
            xlabel.append(j)
            ylabel.append(A)
        cv2.imshow("Imagen",img)
        cv2.waitKey(0)
            
plt.plot(xlabel,ylabel,"bo")
plt.show()
cv2.destroyAllWindows()
databook.close()
