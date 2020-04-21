import cv2
import pytesseract
import numpy as np
import spacy


nlp = spacy.load('en_core_web_md')


def display_images(sample, process=[], resize=True, save=False):
    if resize:
        cv2.namedWindow(sample[0], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(sample[0], 720, 1300)
    cv2.imshow(sample[0], sample[1])

    if save:
        cv2.imwrite('sample.png', sample[1])

    if len(process):
        for i, img in enumerate(process):

            if resize:
                cv2.namedWindow('processed '+str(i+1), cv2.WINDOW_NORMAL)
                cv2.resizeWindow('processed '+str(i+1), 720, 1300)
            cv2.imshow('processed '+str(i+1), img)

            if save:
                cv2.imwrite('processed'+str(i+1)+'.png', img)

    cv2.waitKey()
    cv2.destroyAllWindows()


def image_to_text(image):
    try:
        value = pytesseract.image_to_string(image, lang='eng', config='--oem 3 --psm 6')
        value = value.strip("\n")
    except ValueError as e:
        value = ''

    return value


def get_keyword(word, keywords):
    word = nlp(str(word.lower()))

    similarities = []
    for keyword in keywords:
        keyword = nlp(str(keyword.lower()))
        sm = word.similarity(keyword) if keyword.has_vector and word.has_vector else 0
        similarities.append(sm)

    if max(similarities) > 0.98:
        key = similarities.index(max(similarities))
        return str(key)
    else:
        return False


def dilate_image(image, k=(1, 1), iterations=2):
    kernel = np.ones(k, np.uint8)
    dilation = cv2.dilate(image, kernel, iterations)

    return dilation


def dilate_erosion_image(image):
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    return erosion


def image_to_black_and_white(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (thresh, im_bw) = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    return gray


def detect_hv_lines(bw_image):
    h_img = bw_image.copy()
    v_img = bw_image.copy()

    h_scale = 20
    v_scale = 20

    h_size = int(h_img.shape[1] / h_scale)

    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    h_erode_img = cv2.erode(h_img, h_structure, 1)
    h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)

    v_size = int(v_img.shape[0] / v_scale)

    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_erode_img = cv2.erode(v_img, v_structure, 1)
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

    return h_dilate_img, v_dilate_img


def contours_box(contours, method=None, min_area=False):
    boxes = []

    for cnt in contours:
        if min_area:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue

        if method == "rotated_rectangle":
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box).tolist()
        else:
            box = cv2.boundingRect(cnt)

        boxes.append(box)

    return boxes


def detect_blocks(masked_img):
    height, width = masked_img.shape[0], masked_img.shape[1]
    kh = int((10/2577)*height)
    kw = int((12/1837)*width)
    dilated_image = dilate_image(~masked_img, (kh, kw), 5)

    contours, hierarchy = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(self.image, contours, -1, (0, 255, 0), 3)
    # display_images(["", self.image])

    # Collect all the boxes
    boxes = contours_box(contours, min_area=2000)
    boxes = [[(box[0], box[1]), (box[0] + box[2], box[1] + box[3])] for box in boxes]

    boxes = remove_inner_boxes(boxes)

    return boxes


def remove_inner_boxes(boxes):
    remove_ids = []
    for i, sec1 in enumerate(boxes):
        for j, sec2 in enumerate(boxes):
            if i != j:
                mnx = sec1[0][0] <= sec2[0][0]
                mny = sec1[0][1] <= sec2[0][1]
                mxx = sec1[1][0] >= sec2[1][0]
                mxy = sec1[1][1] >= sec2[1][1]

                if mnx and mny and mxx and mxy:
                    remove_ids.append(j)

    # Removing inside blocks
    remove_ids = np.unique(np.array(remove_ids))
    for index in sorted(remove_ids, reverse=True):
        del boxes[index]

    boxes.reverse()
    return boxes
