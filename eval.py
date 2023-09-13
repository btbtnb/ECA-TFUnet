import numpy as np
import glob
import tqdm
from PIL import Image
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import io
from skimage import measure
from scipy import ndimage
from sklearn.metrics import f1_score
from osgeo import gdal

def ConfusionMatrix(numClass, imgPredict, Label):  

    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  


    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA

def classPixelAccuracy(confusionMatrix):

        classAcc = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
        return classAcc
 
def meanPixelAccuracy(confusionMatrix):
        classAcc = classPixelAccuracy(confusionMatrix)
        meanAcc = np.nanmean(classAcc)
        return meanAcc


  
def Precision(confusionMatrix):  

    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return precision  


def Recall(confusionMatrix):

    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return recall

def mRecall(confusionMatrix):

    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    mRecall = np.nanmean(recall)
    return mRecall


def DSC(confusionMatrix):
  
    intersection = np.diag(confusionMatrix)+np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) 
    DSC = intersection / union
    return DSC

def MDSC(confusionMatrix):

    intersection = 2*np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) 
    DSC = intersection / union
    MDSC = np.nanmean(DSC)
    return MDSC
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score

def mF1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    mF1Score = np.nanmean(f1score)  
    return  mF1Score

def IntersectionOverUnion(confusionMatrix):  

    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  

    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):

    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset 
    return im_proj,im_geotrans,im_width, im_height,im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape 

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def eval_new(label_all,predict_all,classNum):
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()

    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)

    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    acc = classPixelAccuracy(confusionMatrix)
    macc = meanPixelAccuracy(confusionMatrix)
    NDSC    = DSC(confusionMatrix)  
    NMDSC    = MDSC(confusionMatrix)      
    MF1 = mF1Score(confusionMatrix)
    mr =  mRecall(confusionMatrix)

    print("")
    print("confusionMatrix:")
    print(confusionMatrix)
    print("precision:")
    print(precision)
    print("recall:")
    print(recall)
    print("mrecall:")
    print(mr)
    print("F1-Score:")
    print(f1ccore)
    print("mF1-Score:")
    print(MF1)
    print("OA:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)
    print("acc:")
    print(acc)
    print("macc:")
    print(macc)
    print("DSC:")
    print(NDSC)
    print("MDSC:")
    print(NMDSC)
    

if __name__ == "__main__":
    #################################################################

    LabelPath = "./dataset/CamVid/test/labels"
    PredictPath = "./dataset/CamVid/test/pre"
    classNum = 12

    labelList = os.listdir(LabelPath)
    PredictList = os.listdir(PredictPath)

    im_proj,im_geotrans,im_width, im_height, Label0 = read_img(LabelPath + "//" + labelList[0])
    label_num = len(labelList)



    label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    predict_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    for i in range(label_num):
        im_proj,im_geotrans,im_width, im_height, Label = read_img(LabelPath + "//" + labelList[i])
        label_all[i] = Label

        im_proj,im_geotrans,im_width, im_height, Predict = read_img(PredictPath + "//" + PredictList[i])
        predict_all[i] = Predict


    eval_new(label_all,predict_all,classNum)
