import os 
import tensorflow as tf
import matplotlib.pyplot as plt


def read_and_decode(filename, reshape_dims):
 # 1. Read the file.
 img = tf.io.read_file(filename)
 # 2. Convert the compressed string to a 3D uint8 tensor.
 img = tf.image.decode_jpeg(img, channels=3)
 # 3. Convert 3D uint8 to floats in the [0,1] range.
 img = tf.image.convert_image_dtype(img, tf.float32)
 # 4. Resize the image to the desired size.
 return tf.image.resize(img, reshape_dims)



f,ax = plt.subplots(2,5,figsize=(15,15))
id = 0
dir_path_1 = 'Input/'
dir_path_2 = 'Output/'

for file in os.listdir(dir_path_1):
    img_path = os.path.join(dir_path_1,file)
    img = read_and_decode(img_path,[600,500])
    ax[0][id].imshow(img.numpy())
    ax[0][id].axis('off')
    id = id + 1
    if(id==5):  
        break

id = 0 
for file in os.listdir(dir_path_2):
    img_path = os.path.join(dir_path_2,file)
    img = read_and_decode(img_path,[600,500])
    ax[1][id].imshow(img.numpy())
    ax[1][id].axis('off')
    id = id + 1
    if(id==5):  
        break

plt.show()