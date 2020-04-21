import cv2
import os
import json
import numpy as np
import pandas as pd

from .utils import (display_images, detect_hv_lines, image_to_black_and_white,
                    detect_blocks, image_to_text, get_keyword)


class DetectCoordinates:

    def __init__(self, image, display, keyword_pixel, page_num):

        self.idx = page_num
        self.display = display
        self.keyword_pixel = keyword_pixel

        self.image = image
        self.bnw_image = []
        self.h_dilate_img, self.v_dilate_img = [], []
        self.height, self.width = 0, 0

        if self.keyword_pixel:
            keyword_file = os.path.join(os.path.dirname(__file__), 'raw_keywords', 'page'+str(self.idx)+'.json')
        else:
            keyword_file = os.path.join(os.path.dirname(__file__), 'keywords', 'page'+str(self.idx)+'.json')

        data = pd.read_json(keyword_file)
        self.keywords = data.keywords.values.tolist()

    def start(self):
        self.generate_images()

        # Mask all the detected lines
        bw_img = self.bnw_image.copy()
        bw_img[bw_img < 150] = 0
        mask = cv2.add(self.h_dilate_img, self.v_dilate_img)
        # mask[mask != 0] = 255
        masked_img = cv2.add(bw_img, mask)

        if self.display:
            display_images(["Image after removing lines", masked_img], [bw_img])

        # Detect the text blocks
        blocks = detect_blocks(masked_img)
        # Remove the line have less width:
        # print([block[1][0] - block[0][0] for block in blocks])
        blocks = [block for block in blocks if block[1][0] - block[0][0] > 50]
        # Display detected blocks
        if self.display:
            sample = self.image.copy()
            for region in blocks:
                x_min, y_min = region[0][0], region[0][1]
                x_max, y_max = region[1][0], region[1][1]
                cv2.rectangle(sample, (x_min, y_min), (x_max, y_max), 255, 5)

            display_images(["Detected Blocks", sample])

        if self.keyword_pixel:
            corr = self.keyword_corr(blocks)

            with open('Keywords JSON\\page'+str(self.idx)+'.json', 'w') as json_file:
                json.dump(corr, json_file)

            res = False
        else:
            res = self.detect_corr(blocks)

        return res

    def keyword_corr(self, blocks):
        coordinates = []
        words = [key["word"] for key in self.keywords]

        for region in blocks:

            x_min, y_min = region[0][0], region[0][1]
            x_max, y_max = region[1][0], region[1][1]

            text_img = self.bnw_image[y_min:y_max, x_min:x_max]
            text = image_to_text(text_img)
            keyword_id = get_keyword(text, words) if len(text) else False

            if keyword_id:
                keyword = words[int(keyword_id)]
                del words[int(keyword_id)]
                key = keyword

                coordinates.append({
                    "keyword": key,
                    "keyword_x": x_min,
                    "keyword_y": y_min
                })
                # display_images(["", self.image], [text_img], resize=False, save=True)

        return coordinates

    def detect_corr(self, blocks):

        words = [key["word"] for key in self.keywords]
        coordinates = {}
        x_diff, y_diff = [], []
        for region in blocks:
            x_min, y_min = region[0][0], region[0][1]
            x_max, y_max = region[1][0], region[1][1]

            text_img = self.image[y_min:y_max, x_min:x_max]
            text = image_to_text(text_img)
            keyword_id = get_keyword(text, words) if len(text) else False

            if keyword_id:
                keyword = self.keywords[int(keyword_id)]
                self.keywords[int(keyword_id)]["processed"] = True
                key = keyword["word"]
                diff = keyword["diff"]

                x = int(((diff[0]+x_min)/1837)*self.width)  # x_min + diff[0]
                y = int(((diff[1]+y_min)/2577)*self.height)
                w = int((diff[2] / 1837) * self.width)
                h = int((diff[3] / 2577) * self.height)

                # Adding the pixel errors
                x_diff.append(x-keyword["corr"][0])
                y_diff.append(y-keyword["corr"][1])

                xx = x + w
                yx = y + h

                coordinates[key] = [x, y, w, h]
                # print(text)
                val_img = self.image[y:yx, x:xx]
                display_images(["", self.image], [text_img, val_img], resize=False, save=False)

        words = [key["word"] for key in self.keywords]

        x_diff = [err for err in x_diff if -50 < err < 50]
        y_diff = [err for err in y_diff if -50 < err < 50]
        x_err = int(np.mean(x_diff))
        y_err = int(np.mean(y_diff))
        print(x_err)
        print("\tX Error", x_err)
        print(y_err)
        print("\tY Error", y_err)

        for idx, word in enumerate(words):
            keyword = self.keywords[idx]
            if keyword["processed"]:
                continue

            corr = keyword["corr"]
            diff = keyword["diff"]
            key = keyword["word"]
            self.keywords[int(idx)]["processed"] = True

            x = corr[0] + x_err
            y = corr[1] + y_err
            w = int((diff[2] / 1837) * self.width)
            h = int((diff[3] / 2577) * self.height)

            xx = x + w
            yx = y + h

            coordinates[key] = [x, y, w, h]
            # print(key, [x, y, w, h])

            val_img = self.image[y:yx, x:xx]
            display_images(["", self.image], [val_img], resize=False, save=False)

        return coordinates

    def generate_images(self):
        self.bnw_image = image_to_black_and_white(self.image)

        self.h_dilate_img, self.v_dilate_img = detect_hv_lines(~self.bnw_image)
        self.height, self.width = self.image.shape[0], self.image.shape[1]
        print("\tWidth:", self.width)
        print("\tHeight:", self.height)

        if self.display:
            display_images(["page " + str(self.idx + 1), self.image],
                           [self.bnw_image, self.h_dilate_img, self.v_dilate_img], resize=True)