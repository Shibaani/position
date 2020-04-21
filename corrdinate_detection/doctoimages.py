import os
import io
import fitz
import cv2
import numpy as np

from pathlib import Path
from PIL import Image, ImageSequence


from .utils import display_images


# Class to convert the input document to list of images
class FileToImages:

    # Initialize the basic details during creating FileToImages (load_images) class
    # 1. File Location
    # 2. File Name
    # 3. File Extension
    def __init__(self, file_path, output_folder):
        self.file_path = str(Path(file_path))
        self.file_name = os.path.split(file_path)[1].split(".")[0].strip()
        self.file_ext = os.path.split(file_path)[1].split(".")[-1].strip().lower()
        self.output_folder = output_folder

        print("\tFile Path:", self.file_path)
        print("\tFile Name:", self.file_name)
        print("\tFile Type:", self.file_ext)

    # This function will load the input document and return the list of images
    # The return images are in byte format (PIL based), color mode as RGB
    def load(self):
        # Check whether the input file is acceptable or not and adding extension to variable
        if self.file_ext not in ["pdf", "tiff", "tif", "jpg", "jpeg", "png"]:
            raise TypeError("Invalid file format. Only pdf, tiff, tif, jpg, jpeg, png is acceptable")

        ext = self.file_ext

        # Check the type of document and send to respective function to extract images from document
        if ext in ["pdf"]:
            images = self.images_from_pdf()
        elif ext in ["tiff", "tif"]:
            images = self.images_from_tiff()
        else:
            images = [Image.open(self.file_path).convert('RGB')]

        print("\tNumber of Pages:", len(images))

        for idx, image in enumerate(images):
            image = np.array(image)
            image_file = os.path.join(self.output_folder, "page"+str(idx+1)+'.png')
            cv2.imwrite(image_file, image)

        return images

    # Extract the images from pdf using fitz
    def images_from_pdf(self):

        images = []
        doc = fitz.open(self.file_path)
        zoom_x = 1
        zoom_y = 1
        for page in doc:
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.getPixmap(matrix=mat, alpha=False).getImageData()
            image = Image.open(io.BytesIO(pix))
            image = np.array(image)

            if len(str(image.size)) <= 7:
                mat = fitz.Matrix(3, 3)
                pix = page.getPixmap(matrix=mat, alpha=False).getImageData()
                image = Image.open(io.BytesIO(pix))
                image = np.array(image)

            image = Image.fromarray(image)
            images.append(image.convert('RGB'))

        return images

    # Extract the images from tiff file using PIL
    def images_from_tiff(self):
        images = []

        for i, image in enumerate(ImageSequence.Iterator(Image.open(self.file_path))):
            byte_array = io.BytesIO()
            image.save(byte_array, format='PNG')
            image = Image.open(io.BytesIO(byte_array.getvalue()))
            images.append(image.convert('RGB'))

        return images
