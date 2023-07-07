""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import random
import fire
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

def get_value_from_txt(path_txt):
    if os.path.exists(path_txt):
        with open(path_txt,'r',encoding="utf-8") as f:
            data=f.read()
            data=data.replace('"','')
            f.close()
            return data
    print("khong ton tai : ",path_txt)
    return ""
def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(List_inputPath, outputPath, checkValid=True):
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    max_length=0
    kytu=''
    list_len=[]
    list_data=[]
    tongw=0
    tongh=0
    tong=0

    os.makedirs(outputPath, exist_ok=True)
    # with open(gtFile, 'r', encoding='utf-8') as data:
    #     datalist = data.readlines(

    for number in range(len(List_inputPath)):
        inputPath=List_inputPath[number]
        folder_txt=List_folder_txt[number]

        list_img=[i for i in os.listdir(inputPath) if i.endswith(".jpg")]
        random.shuffle(list_img)
        nSamples = len(list_img)
        tong+=nSamples
        for i in list_img:
            img=cv2.imread(inputPath+i)
            h, w, c = img.shape
            tongh+=h
            tongw+=w
            cv2.destroyAllWindows()
     
            path_txt=folder_txt+i.replace(".jpg",".txt")
            label=get_value_from_txt(path_txt)
            imagePath = inputPath+i
            if not label:
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

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    str_list_data="["
    for i in list_data:
        str_list_data+="["+str(i[0])+","+str(i[1])+"],"
    str_list_data=str_list_data[0:len(str_list_data)-1]+"]"
    with open(outputPath+"/outdaset.txt", "w", encoding='utf8') as f:
        f.write(str(max_length)+"\n H :"+str(tongh/tong)+"\n w :"+str(tongw/tong)+'\n'+str_list_data+"\n"+kytu)
        f.close()

    print('Created dataset with %d samples' % nSamples)
    sys.exit()
"CUDA_VISIBLE_DEVICES=1 python3 create_lmdb_dataset_AnhTH_name_img.py"
if __name__ == '__main__':
    # List_inputPath=[
    #     "/home/vbpo/SV7_SGC_HDDT_Img/HA_BK/Data_OCR/test/train_value_txt/img/",
    #     "/home/vbpo/SV7_SGC_HDDT_Img/HA_BK/Data_OCR/test/train_value_txt1/img/"
    #     ] 
    # List_folder_txt=[
    #     "/home/vbpo/SV7_SGC_HDDT_Img/HA_BK/Data_OCR/test/train_value_txt/txt/",
    #     "/home/vbpo/SV7_SGC_HDDT_Img/HA_BK/Data_OCR/test/train_value_txt1/txt/"
    #     ]
    # outputPath="data_train/test/train_value_txt"    

    List_inputPath=[
        "data_train/train_thu_chuviettay/img/"
        ] 
    List_folder_txt=[
        "data_train/train_thu_chuviettay/txt/"
        ]
    outputPath="data_train/train_thu_chuviettay/db_train"    
    fire.Fire(createDataset(List_inputPath=List_inputPath,outputPath=outputPath))
