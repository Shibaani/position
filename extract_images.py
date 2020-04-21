import os
from corrdinate_detection import FileToImages

if __name__ == "__main__":

    input_file = "D:\\SCB-Poject\\Workspace\\Experiments\\Singapore\\data\\AOF_BankingBill Priority_Final.pdf"
    output_folder = os.path.join(os.getcwd(), "pages")

    print("Extracting the images from document.")
    images = FileToImages(input_file, output_folder).load()

    print("Extraction is completed and images stored at", output_folder)