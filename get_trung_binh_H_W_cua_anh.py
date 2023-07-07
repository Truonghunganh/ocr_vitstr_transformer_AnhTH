import cv2
import os
import sys
list_folder=["data_train/bienso/new_lp_data_1dong/train/","data_train/bienso/new_lp_data_1dong/test/"]
list_folder=['data_train/train_thu_chuviettay/img/']
tongw=0
tongh=0
tong=0
vt=0

for folder_read in list_folder:
	list_img=[item for item in os.listdir(folder_read) if item.lower().endswith('.jpg') ]
	# list_img=[item for item in os.listdir(folder_read) if item.lower().endswith('.png') ]
	print(len(list_img))
	tong+=len(list_img)
	for item in list_img:
		sys.stdout.write('%d img \r' % (vt,))
		sys.stdout.flush()
		vt+=1
		img=cv2.imread(folder_read+item)
		h, w, c = img.shape
		tongh+=h
		tongw+=w
		cv2.destroyAllWindows()
print('h : ',tongh/tong)
print('w : ',tongw/tong)
'''
detect_CCPD2019_model783anhtrain_320756: 
	h:114.38052341535175
	w:249.94094934386356

Anhth_Bien_so_xe : TPS-ResNet-BiLSTM-Attn-Seed30576
h :  79.78452458360427
w :  202.27424598609514 

class04:TPS-ResNet-BiLSTM-Attn-Seed8764
h :  68.35926502981786
w :  341.0286538082702

/home/vbpo/Desktop/AnhTH/yolov5/data/train/bienso/data_train_yolo_kitu_biensoxe/images/train/
h :  345.1543230016313
w :  578.4910277324633

diem cua yobe:
h :  213.32994890696668
w :  127.27960198320737
'''
# PO :
	#Class01: TPS-ResNet-BiLSTM-Attn-Seed32242
	   # 48.53501883026786 :48
	   # 319.1990677137799 :320

	#Class02:TPS-ResNet-BiLSTM-Attn-Seed3103
	   # 48.92539365832614 : 48
	   # 191.35358471742882: 192
	#Class03:TPS-ResNet-BiLSTM-Attn-Seed7637
	   # 186.8104855105858 : 186
	   # 633.7184797636152 : 634

"""
	HD: 
		class05 167000
			68.09983056225789   : 86
			296.52527496213094	: 296
			myCrop
			58.10508428006546   64
			286.78453332377643	320
"""

# class02: TPS-ResNet-BiLSTM-Attn-Seed12851
#	67.66444305130344 : 64
#	457.51279086045395: 448

# class04: TPS-ResNet-BiLSTM-Attn-Seed30629
#	63.10721039117958 : 64
#	321.16625283189853: 320

# class05 : 2320
#	67.94391780319246 : 68
#	293.3969787780564 : 293
# class08 :12077
# 65.97465803377074	: 64
# 466.39286171592727	: 448
# 64
# 448


# class08_5x6_90pt :48b la TPS-ResNet-BiLSTM-Attn-Seed7748
# class08_5x6 :48b la TPS-ResNet-BiLSTM-Attn-Seed9257
"""
bk:
	class01:
		h : 83.56289247601804   : 84
		W : 321.15290673036174	: 320
		h-m : 83.56847097924492 : 84
		w_m : 318.3010957433918 : 320
		81.50010281719103: 82
		322.645006511755 :322
"""