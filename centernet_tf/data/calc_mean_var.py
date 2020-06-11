import os
from PIL import Image
import skimage
import numpy as np

dataset_path = "/ssd1/liuxing/project/centerNet/centerNet/CenterNet-master/data/wine/data/images/"
logo_dirname = ["贵州茅台酒", "西凤酒"]
#dir_path = os.listdir(dataset_path)

size = 512
image_data = []
for logo_dir in logo_dirname:
    logo_dirpath = os.path.join(dataset_path, logo_dir)
    dir_path = os.listdir(logo_dirpath)
    for dirname in dir_path:
        filesname = os.path.join(dataset_path, logo_dir, dirname)
        if not os.path.isdir(filesname):
            try:
                image = skimage.io.imread(filesname)
                if not image.shape[-1] == 3:
                    continue
            except Exception as e:
                print(e)
                os.remove(dirname)
            image = skimage.transform.resize(image, (size, size))
            image_data.append(image)
        else:
            files = os.listdir(filesname)
            for imgfile in files:
                imgfilename = os.path.join(dataset_path, dirname, imgfile)
                try:
                    image = skimage.io.imread(imgfilename)
                    if not image.shape[-1] == 3:
                        continue
                except Exception as e:
                    print(e)
                    os.remove(imgfilename)
                image = skimage.transform.resize(image, (size, size))
                image_data.append(image)
image_array = np.array(image_data)
print(image_array.shape) #(8963, 512, 512, 3)
image_mean = np.mean(image_array, axis=(0,1,2))
print("mean shape:{}\tmean:{}".format(image_mean.shape, image_mean))
image_var = np.var(image_array, axis=(0,1,2))
print("variance shape:{}\tvariance:{}".format(image_var.shape, image_var))

#nettool
#(8963, 512, 512, 3)
#mean shape:(3,) mean:[0.58599497, 0.55584838, 0.54897476]
#variance shape:(3,)     variance:[0.13181064, 0.12557935, 0.13066235]

#wine
#(1355, 512, 512, 3)
#mean shape:(3,) mean:[0.68719612 0.56187135 0.51211851]
#variance shape:(3,)     variance:[0.09315962 0.12737098 0.13884356]
