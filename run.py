import cv2
from corrdinate_detection import DetectCoordinates

if __name__ == "__main__":

    input_file = "D:\\SCB-Poject\\Workspace\\Experiments\\Singapore\\pages\\page2.png"

    print("Detect and extract the coordinates")
    image = cv2.imread(input_file)
    data = DetectCoordinates(image, display=True, keyword_pixel=False, page_num=1).start()
    if data:
        print(data)
    print("Extraction Completed")
