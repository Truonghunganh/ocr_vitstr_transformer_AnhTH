""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import random
import fire
import re
import os
import lmdb
import cv2
import sys
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    # env = lmdb.open(outputPath, map_size=1099511627776)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    max_length=0
    kytu=''
    list_len=[]
    list_data=[]
    list_value=[]
    # loai_img=".png"

    loai_img=".jpg"
    tongw=0
    tongh=0
    tong=0
    wmax=0
    hmax=0
    list_img=[i for i in os.listdir(inputPath) if i.endswith(loai_img)]
    random.shuffle(list_img)
    nSamples = len(list_img)
    print(nSamples)

    tong=nSamples
    for i in list_img:
        label=i.split(loai_img)[0].split("_")[1]
        img=cv2.imread(inputPath+i)
        h, w, c = img.shape
        if hmax<h: hmax=h
        if wmax<w:wmax=w
        tongh+=h
        tongw+=w
        cv2.destroyAllWindows()
        # if "$" in label:
        
        #     label=label.replace('$','')
        # label=label.lower()
        kitunhan=label[0]
        # kitu=kitu[-1]
        kiemtra=False
        for so in range(len(list_value)):
            if kitunhan==list_value[so][0]:
                kiemtra=True
                list_value[so][1]+=1
                break
        if kiemtra==False:
            list_value.append([kitunhan,1])
        imagePath = inputPath+i
        if not label:
            print("label not")
            continue
        length=len(label)
        if length not in list_len:
            list_len.append(length)
            list_data.append([length,1])
        else:
            for number in range(len(list_data)):
                if length ==list_data[number][0]:
                    list_data[number][1]+=1
                    break
        if length>max_length:
            max_length=length
        for j  in label:
            if j not in kytu:
                kytu+=j
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            f.close()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        # cache[labelKey] = label

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(label,'Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    listchar=sorted(kytu)
    kytu=''
    for i in listchar:
        kytu+=i
    str_list_data="["
    for i in list_data:
        str_list_data+="["+str(i[0])+","+str(i[1])+"],"
    str_list_data=str_list_data[0:len(str_list_data)-1]+"]"
    str_list_data+="\n value : ["
    for i in list_value:
        str_list_data+="["+str(i[0])+","+str(i[1])+"],"
    str_list_data=str_list_data[0:len(str_list_data)-1]+"]"
    with open(outputPath+"/outdaset.txt", "w", encoding='utf8') as f:
        f.write(str(max_length)+"\n H :"+str(tongh/tong)+", hmax="+str(hmax)+"\n w :"+str(tongw/tong)+", wmax="+str(wmax)+"\n"+kytu+"\n"+str_list_data)
        f.close()

    print('Created dataset with %d samples' % nSamples,label)
    sys.exit()
"CUDA_VISIBLE_DEVICES=1 python create_lmdb_dataset_AnhTH_name_img.py"
"CUDA_VISIBLE_DEVICES=0 python create_lmdb_dataset_AnhTH_name_img.py"
if __name__ == '__main__':
    # inputPath="data_train/test/img/" 
    # outputPath="data_train/test/db_train/" 
    inputPath="/home/vbpo/Desktop/AnhTH/roatienza_deep-text-recognition-benchmark/data_train/SGC/HD/Class08_V1/Class08_ALL/" 
    outputPath="data_train/SGC/HD/Class08_V1/db_train/" 
    fire.Fire(createDataset(inputPath=inputPath,outputPath=outputPath))
