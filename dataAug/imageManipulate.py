# import the Python Image processing Library

from PIL import Image
import os
from matplotlib import pyplot as plt
import albumentations as A
import cv2





dir_path ="raw_images/"
processed_dir_path="augmented_pics/"
augmentation_dir_path="ag_pics/"



def simpleTransforms(path):
# Create an Image object from an Image
        colorImage  = Image.open(os.path.join(dir_path, path))
        # Rotate it by 45 degrees
        rotated     = colorImage.rotate(45)
        # Rotate it by 90 degrees
        transposed  = colorImage.transpose(Image.ROTATE_90)
        # Do a flip of left and right
        flippedImage = colorImage.transpose(Image.FLIP_LEFT_RIGHT)
        # Display the Original Image
        #colorImage.show()
        # Display the Image rotated by 45 degrees
        #rotated.show()
        rotated.save(processed_dir_path + "rot_"+ path)
        # Display the Image rotated by 90 degrees
        #transposed.show()
        transposed.save(processed_dir_path + "transp_" + path)
        #flippedImage.show()
        flippedImage.save(processed_dir_path + "flipped_" + path)



def albumentationTransforms(path):
        # Declare an augmentation pipeline
        transform = A.Compose([
            #A.RandomCrop(width=512, height=512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])
        # Read an image with OpenCV and convert it to the RGB colorspace
        image = cv2.imread(os.path.join(dir_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        #transformed_image.save(augmentation_dir_path + "augmt" + path)
        im = Image.fromarray(transformed_image)
        im.save(augmentation_dir_path + "augmt" + path)

for path in os.listdir(dir_path):
    if not os.path.exists(processed_dir_path):
        os.makedirs(processed_dir_path)
    if not os.path.exists(augmentation_dir_path):
        os.makedirs(augmentation_dir_path)


    if os.path.isfile(os.path.join(dir_path, path)):

        simpleTransforms(path)
        albumentationTransforms(path)
        
        

        

