import cv2
import random as ran

org_lable = cv2.imread("org_lable.png")
org_lable = org_lable[:,:,2]
org_img = cv2.imread("org_img.png")

h = org_img.shape[0]
w = org_img.shape[1]

slice_size_i = 360
slice_size_j = 480
nb_train = 400
nb_test = 400

w_s = w - slice_size_j
h_s = h - slice_size_j

for i in range(nb_train + nb_test):
    x = int(ran.random()*w_s)
    y = int(ran.random()*h_s)

    sl_lable = org_lable[y:y+slice_size_i, x:x+slice_size_j]
    sl_img = org_img[y:y+slice_size_i, x:x+slice_size_j]

    cv2.imwrite(r'./lable/'+str(i)+r'.png', sl_lable)
    cv2.imwrite(r'./img/'+str(i)+r'.png', sl_img)
    # print(x,y)

print("数据切割完毕！")
