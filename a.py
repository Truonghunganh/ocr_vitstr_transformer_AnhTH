import os,shutil
# label="ạĩộsốĐờìẻảKHÁIQUVỀBỂNẢOỆAMồvếăúâểặẫeé8"
# labels=sorted(label)
# label=""
# for i in labels:
#     label+=i
# # label=label.encode()
# label='oll.png'
# print(label[:-4])
# name_folder="/home/vbpo-100367/SV7_DRI_DATA/AnhTH/Du_lieu_yolo_cua_du_an/data_chu_viet_tay/ocr_dataset/"

# list_duoi=[]
# for folder in os.listdir(name_folder):
#     for i in os.listdir(name_folder+folder):
#         if i[-4:] =='.png':
#             shutil.move(name_folder+folder+"/"+i,name_folder+folder+"/"+i.replace(".png",".jpg"))
#         if i[-4:] not in list_duoi:
#             list_duoi.append(i[-4:])
# print(list_duoi)
a=" !#$%&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}°²ÀÁÂÃÈÉÊÌÍÐÒÓÔÕÖÙÚÜÝàáâãèéêìíðòóôõöùúüýĀāĂăĐđĨĩŌōŨũŪūƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–—’“”…€™−"
a='…€™−'
print(len(a))
for i in range(len(a)):
    for j in range(i+1,len(a)):
        print(i,j)