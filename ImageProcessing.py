from PIL import Image, ImageFilter
import cv2
import numpy as np
import os

class ImageProcessing:
    def __init__(self):
        self.digit_templates = {
            '0': 'template/z26.png',
            '1': 'template/z27.png',
            '2': 'template/z28.png',
            '3': 'template/z29.png',
            '4': 'template/z30.png',
            '5': 'template/z31.png',
            '6': 'template/z32.png',
            '7': 'template/z33.png',
            '8': 'template/z34.png',
            '9': 'template/z35.png',
            'A': 'template/z0.png',
            'B': 'template/z1.png',
            'C': 'template/z2.png',
            'D': 'template/z3.png',
            'E': 'template/z4.png',
            'F': 'template/z5.png',
            'G': 'template/z6.png',
            'H': 'template/z7.png',
            'I': 'template/z8.png',
            'J': 'template/z9.png',
            'K': 'template/z10.png',
            'L': 'template/z11.png',
            'M': 'template/z12.png',
            'N': 'template/z13.png',
            'O': 'template/z14.png',
            'P': 'template/z15.png',
            'Q': 'template/z16.png',
            'R': 'template/z17.png',
            'S': 'template/z18.png',
            'T': 'template/z19.png',
            'U': 'template/z20.png',
            'V': 'template/z21.png',
            'W': 'template/z22.png',
            'X': 'template/z23.png',
            'Y': 'template/z24.png',
            'Z': 'template/z25.png'
        }

    def calculate_mse(self, img1, img2):
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        return err

    def find_most_similar_digit(self, input_img, digit_templates):
        input_img_gray = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
        min_mse = float('inf')
        most_similar_digit = None
        for digit, template_path in digit_templates.items():
            template_img_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            mse = self.calculate_mse(input_img_gray, template_img_gray)
            if mse < min_mse:
                min_mse = mse
                most_similar_digit = digit
        return most_similar_digit

    def process_image_and_find_digit(self, input_image_path):
        result = self.find_most_similar_digit(input_image_path, self.digit_templates)
        print(f"The most similar digit is: {result}")

    def convert_to_grayscale(self, input_path, output_path):
        image = Image.open(input_path)
        grayscale_image = image.convert("L")
        grayscale_image.save(output_path)

    def median_gaussian(self, image_path, output_path):
        image = Image.open(image_path)
        median_image = image.filter(ImageFilter.MedianFilter())
        median_image.save(output_path)

    def sharpen_filter(self, input_path, output_path):
        image = Image.open(input_path)
        sharpened_image = image.filter(ImageFilter.SHARPEN)
        sharpened_image.save(output_path)

    def denoise_colored_image(self, input_path, output_path):
        colored_image = cv2.imread(input_path)
        denoised_image = cv2.fastNlMeansDenoisingColored(colored_image, None, h=12, hColor=10, templateWindowSize=7, searchWindowSize=21)
        cv2.imwrite(output_path, denoised_image)

    def binary_threshold(self, input_path, output_path, threshold_value=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(gray_image, threshold_value, max_value, threshold_type)
        cv2.imwrite(output_path, binary_image)

    def erode_image(self, input_path, output_path, kernel_size=3, iterations=1):
        binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_image = cv2.erode(binary_image, kernel, iterations=iterations)
        cv2.imwrite(output_path, eroded_image)

    def dilate_image(self, input_path, output_path, kernel_size=3, iterations=1):
        binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=iterations)
        cv2.imwrite(output_path, dilated_image)

    def sobel_filter(self, input_path, output_path):
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude)
        cv2.imwrite(output_path, gradient_magnitude)

    def find_y_pixel_regions(self, input_path):
        img = Image.open(input_path)
        width, height = img.size
        img_data = list(img.getdata())
        white_pixel_count_per_row = []
        for y in range(height):
            row_pixels = img_data[y * width: (y + 1) * width]
            white_pixel_count = sum(1 for pixel in row_pixels if pixel == 255)
            white_pixel_count_per_row.append(white_pixel_count)
        regions = []
        start = None
        for i, count in enumerate(white_pixel_count_per_row):
            if count > 0 and start is None:
                start = i
            elif count == 0 and start is not None:
                end = i - 1
                regions.append((start, end))
                start = None
        if start is not None:
            regions.append((start, len(white_pixel_count_per_row) - 1))
        for region in regions:
            return region

    def find_x_pixel_regions(self, input_path):
        img = Image.open(input_path)
        width, height = img.size
        img_data = list(img.getdata())
        white_pixel_count_per_column = []
        for x in range(width):
            column_pixels = img_data[x::width]
            white_pixel_count = sum(1 for pixel in column_pixels if pixel == 255)
            white_pixel_count_per_column.append(white_pixel_count)
        regions = []
        start = None
        for i, count in enumerate(white_pixel_count_per_column):
            if count > 0 and start is None:
                start = i
            elif count == 0 and start is not None:
                end = i - 1
                regions.append((start, end))
                start = None
        if start is not None:
            regions.append((start, len(white_pixel_count_per_column) - 1))
        for idx, region in enumerate(regions):
            start, end = region
            self.x_crop_image(input_path, f"separate/x{idx}.png", start, end)
            region = self.find_y_pixel_regions(f"separate/x{idx}.png")
            self.crop_and_save_region("preprocessing/st7.png", f"separate/y{idx}.png", start, end, region[0], region[1])
            self.resize_image(f"separate/y{idx}.png", f"separate/z{idx}.png")
            self.binary_threshold(f"separate/z{idx}.png", f"separate/z{idx}.png")

    def x_crop_image(self, input_path, output_path, start, end):
        img = Image.open(input_path)
        cropped_img = img.crop((start, 0, end, img.size[1]))
        cropped_img.save(output_path)

    def y_crop_image(self, input_path, output_path, start, end):
        img = Image.open(input_path)
        cropped_img = img.crop((0, start, img.size[0], end))
        cropped_img.save(output_path)

    def crop_and_save_region(self, input_path, output_filename, start_x, end_x, start_y, end_y):
        img = Image.open(input_path)
        cropped_x = img.crop((start_x, 0, end_x, img.size[1]))
        cropped_region = cropped_x.crop((0, start_y, cropped_x.width, end_y))
        cropped_region.save(output_filename)
        return output_filename

    def resize_image(self, input_path, output_path, target_size=(30, 30)):
        original_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(original_img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)
        self.process_image_and_find_digit(output_path)

    def crop_top_and_bottom_rows(self, input_path, output_path):
        image = cv2.imread(input_path)
        height, width = image.shape[:2]
        start_row = 3
        end_row = height - 3
        start_col = 3
        end_col = width - 3
        cropped_image = image[start_row:end_row, start_col:end_col]
        cv2.imwrite(output_path, cropped_image)

    def color_inversion(self, input_path, output_path):
        image = cv2.imread(input_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(binary_image)
        cv2.imwrite(output_path, inverted_image)

    def delete_all_files_in_folder(self, folder_path):
        try:
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
            print(f"All files in '{folder_path}' deleted successfully.")
        except FileNotFoundError:
            print(f"Folder '{folder_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def main(input_image_path,separate = False):
        image_processor = ImageProcessing()
        image_processor.denoise_colored_image(input_image_path, "preprocessing/st1.png")
        image_processor.convert_to_grayscale("preprocessing/st1.png", "preprocessing/st2.png")
        image_processor.sharpen_filter("preprocessing/st2.png", "preprocessing/st3.png")
        image_processor.binary_threshold("preprocessing/st3.png", "preprocessing/st4.png")
        image_processor.color_inversion("preprocessing/st4.png", "preprocessing/st5.png")
        image_processor.median_gaussian("preprocessing/st5.png", "preprocessing/st6.png")
        image_processor.color_inversion("preprocessing/st6.png", "preprocessing/st7.png")
        image_processor.find_x_pixel_regions("preprocessing/st6.png")
        if(separate == False):
            image_processor.delete_all_files_in_folder("separate")
            image_processor.delete_all_files_in_folder("preprocessing")

