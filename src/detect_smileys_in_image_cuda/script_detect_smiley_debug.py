import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def resize_image(image, scale_percent):
    if scale_percent == 100:
        return image
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    interpolation_method = cv2.INTER_LINEAR
    resized_grayscale_image = cv2.resize(image, dim, interpolation=interpolation_method)
    return resized_grayscale_image




def create_image_with_templates(rgb_image, orig_rgba_template, start_points):
    template_images = get_all_template_images(orig_rgba_template)

    rgba_template = template_images[0]
    blue_channel, green_channel, red_channel, alpha_channel = cv2.split(rgba_template)
    mask = alpha_channel > 0

    template_rows, template_cols, _ = rgba_template.shape
    rgb_image_with_template = rgb_image.copy()



    num_of_points = start_points.shape[0]

    for i in range(num_of_points):
        start_row = start_points[i, 1]
        start_column = start_points[i, 0]

        end_row = start_row + template_rows
        end_column = start_column + template_cols

        rgb_patch = rgb_image_with_template[start_row:end_row, start_column:end_column, :]

        for current_channel in range(0, 3):
            template_current_channel = rgba_template[:, :, current_channel]
            patch_current_channel = rgb_patch[:, :, current_channel]
            patch_with_template_current_channel = np.where(mask, template_current_channel, patch_current_channel)
            rgb_patch[:, :, current_channel] = template_current_channel
            david = 4
        rgb_image_with_template[start_row:end_row, start_column:end_column, :] = rgb_patch
    #bgr_image_with_template = cv2.cvtColor(rgb_image_with_template, cv2.COLOR_RGB2BGR)

    return rgb_image_with_template

def calc_filter_points(points, threshold):
    filtered_points = []

    for point in points:
        if all(np.linalg.norm(point - fp) >= threshold for fp in filtered_points):
            filtered_points.append(point)

    np_filtered_points = np.array(filtered_points)
    return np_filtered_points

def get_all_template_images(template):
    template_images = []
    scale_percents = np.array([100, 50, 25])
    axes = [1, 0, 2]
    for single_scale_percent in scale_percents:
        resized_template = resize_image(image=template, scale_percent=single_scale_percent)
        resized_template_rotate_90 = np.transpose(a=resized_template, axes=axes)
        resized_template_rotate_180 = np.flip(resized_template, axis=0)
        resized_template_rotate_270 = np.transpose(a=resized_template_rotate_180, axes=axes)
        template_images.append(resized_template)
        template_images.append(resized_template_rotate_90)
        template_images.append(resized_template_rotate_180)
        template_images.append(resized_template_rotate_270)
    return template_images

def sumsqdiffr(image, rgba_template, valid_mask=None):
    # template = cv2.cvtColor(rgba_template, cv2.COLOR_RGBA2RGB)
    # if valid_mask is None:
    #     valid_mask = np.ones_like(template)
    # total_weight = valid_mask.sum()

    # image_height = image.shape[0]
    # image_width = image.shape[1]

    # template_height = template.shape[0]
    # template_width = template.shape[1]

    image_height = 6
    image_width = 9

    image_1D = np.arange(1, image_height * image_width + 1)
    image_2d = np.reshape(image_1D, [image_height, image_width])
    image = cv2.merge((image_2d, image_2d, image_2d))

    template_height = 4
    template_width = 5

    template_1D = np.arange(1, template_height * template_width + 1)
    template_2d = np.reshape(template_1D, [template_height, template_width])
    template = cv2.merge((template_2d, template_2d, template_2d))

    if valid_mask is None:
        valid_mask = np.ones_like(template)
    total_weight = valid_mask.sum()

    result_height = image_height - template_height + 1
    result_width = image_width - template_width + 1

    result = np.zeros((result_height, result_width))




    for tr in range(template_height):
        for i in range(tr, result_height + tr):
            for j in range(result_width):
                sample = image[i, j:j + template_width]
                template_tr = template[tr]
                diff_vals = template_tr - sample
                dist = diff_vals ** 2
                result[i - tr, j] += (dist * valid_mask[tr]).sum()
    return result


def match_template_sqdiff_normed_without_loops(image, template, mask=None):
    img_h, img_w = image.shape
    tpl_h, tpl_w = template.shape

    # Compute the squared differences
    squared_diff = (image.astype(np.float32) - template.astype(np.float32)) ** 2

    # Apply mask if provided
    if mask is not None:
        mask_normalized = mask / 255.0
        squared_diff *= mask_normalized

    # Sum the squared differences using filter2D
    result = cv2.filter2D(squared_diff, -1, np.ones(template.shape, dtype=np.float32))

    # Normalize by the number of pixels in the template
    if mask is not None:
        mask_area = cv2.filter2D(mask_normalized, -1, np.ones(template.shape, dtype=np.float32))
        result /= mask_area
    else:
        result /= (tpl_h * tpl_w)

    # Normalize the result to the range [0, 1]
    cv2.normalize(result, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return result

def match_template_sqdiff_normed(image, template, mask=None):
    img_h, img_w = image.shape
    tpl_h, tpl_w = template.shape
    result = np.zeros((img_h - tpl_h + 1, img_w - tpl_w + 1), dtype=np.float32)

    for y in range(result.shape[0]):
        print(f'{y + 1} out of {result.shape[0]}')
        for x in range(result.shape[1]):
            patch = image[y:y + tpl_h, x:x + tpl_w]
            if mask is not None:
                diff = (patch - template) * (mask / 255)
            else:
                diff = patch - template
            sqdiff = np.sum(diff ** 2)
            norm_sqdiff = sqdiff / (tpl_h * tpl_w)
            result[y, x] = norm_sqdiff

    # Normalize the result
    cv2.normalize(result, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return result

def calc_TM_SQDIFF_NORMED(image, template):
    matrix_type = np.float64

    image = image.astype(matrix_type)
    template = template.astype(matrix_type)

    image_height = image.shape[0]
    image_width = image.shape[1]
    template_height = template.shape[0]
    template_width = template.shape[1]

    result_height = image_height - template_height + 1
    result_width = image_width - template_width + 1

    shape = (result_height, result_width, template_height, template_width)
    image_strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    multiple_patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=image_strides)

    result_matrix = np.zeros((result_height, result_width), matrix_type)
    sqr_template = template ** 2
    sum_sqr_template = np.sum(sqr_template)
    for i in range(result_height):
        print(f'{i + 1} out of {result_height}')
        for j in range(result_width):
            patch1 = image[i:i + template_height, j:j + template_width]
            patch = multiple_patches[i, j, :, :]
            diff_matrix = template - patch
            sqr_diff_matrix = diff_matrix ** 2
            nominator = np.sum(sqr_diff_matrix)

            sqr_patch = patch ** 2
            sum_sqr_patch = np.sum(sqr_patch)
            sqr_denominator = sum_sqr_template * sum_sqr_patch
            denominator = np.sqrt(sqr_denominator)

            result_current_value = nominator / denominator
            result_matrix[i, j] = result_current_value

    return result_matrix

# def calc_TM_SQDIFF_NORMED(image, template):
#     matrix_type = np.float64
#     image = image.astype(matrix_type)
#     template = template.astype(matrix_type)
#
#     image_height = image.shape[0]
#     image_width = image.shape[1]
#     template_height = template.shape[0]
#     template_width = template.shape[1]
#
#     result_height = image_height - template_height + 1
#     result_width = image_width - template_width + 1
#     shape = (result_height, result_width, template_height, template_width)
#     image_strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
#     multiple_patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=image_strides)
#
#     template_strides = (0, 0, template.strides[0], template.strides[1])
#     multiple_templates = np.lib.stride_tricks.as_strided(template, shape=shape, strides=template_strides)
#
#     diff_matrix = multiple_templates - multiple_patches
#     sqr_diff_matrix = diff_matrix ** 2
#     nominators_matrix = np.sum(sqr_diff_matrix, axis=(2, 3))
#
#     sqr_template = template ** 2
#     sum_sqr_template = np.sum(sqr_template)
#
#     sqr_multiple_patches = multiple_patches ** 2
#     sum_sqr_patches = np.sum(sqr_multiple_patches, axis=(2, 3))
#
#     sqr_denominators_matrix = sum_sqr_template * sum_sqr_patches
#     denominators_matrix = np.sqrt(sqr_denominators_matrix)
#
#     result_matrix = nominators_matrix / denominators_matrix
#
#     david = 5
#
#     return result_matrix

# def calc_TM_SQDIFF_NORMED(image, template):
#     reference_matchTemplate = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
#     image_height = image.shape[0]
#     image_width = image.shape[1]
#     template_height = template.shape[0]
#     template_width = template.shape[1]
#
#     result_height = image_height - template_height + 1
#     result_width = image_width - template_width + 1
#
#     result_float = np.zeros((result_height, result_width), np.float64)
#
#     template_float64 = template.astype(np.float64)
#     image_float64 = image.astype(np.float64)
#
#     for result_i in range(0, result_height):
#         print(f'{result_i + 1} out of {result_height}')
#         for result_j in range(0, result_width):
#
#             image_patch_float64 = image_float64[result_i:result_i+template_height, result_j:result_j+template_width]
#             diff_matrix = template_float64 - image_patch_float64
#             sqr_diff_matrix = np.multiply(diff_matrix, diff_matrix)
#             current_nominator_float_64 = np.sum(sqr_diff_matrix)
#
#             sqr_template_float64 = template_float64 * template_float64
#             sum_sqr_template = np.sum(sqr_template_float64)
#
#             sqr_image_patch_float64 = image_patch_float64 * image_patch_float64
#             sum_sqr_patch = np.sum(sqr_image_patch_float64)
#
#             sqr_denominator = sum_sqr_template * sum_sqr_patch
#             denominator = np.sqrt(sqr_denominator)
#
#             current_val_float_64 = current_nominator_float_64 / denominator
#             #current_val_float_64 = current_nominator_float_64
#
#             current_val_float_32 = np.float32(current_val_float_64)
#             result_float[result_i, result_j] = current_val_float_64
#
#             reference_val_float_32 = reference_matchTemplate[result_i, result_j]
#
#             c = np.float64(661466430) - np.float64(661466400)
#
#             diff_from_reference = current_val_float_32 - reference_val_float_32
#             sqr_diff_from_reference = diff_from_reference * diff_from_reference
#             david = 5
#
#
#     return result_float


def calc_TM_SQDIFF(image, template):
    reference_matchTemplate = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    image_height = image.shape[0]
    image_width = image.shape[1]
    template_height = template.shape[0]
    template_width = template.shape[1]

    result_height = image_height - template_height + 1
    result_width = image_width - template_width + 1

    result_float_32 = np.zeros((result_height, result_width), np.float32)

    template_float64 = template.astype(np.float64)
    image_float64 = image.astype(np.float64)

    for result_i in range(0, result_height):
        print(f'{result_i + 1} out of {result_height}')
        for result_j in range(0, result_width):

            image_patch_float64 = image_float64[result_i:result_i+template_height, result_j:result_j+template_width]
            diff_matrix = template_float64 - image_patch_float64
            sqr_diff_matrix = np.multiply(diff_matrix, diff_matrix)
            current_val_float_64 = np.sum(sqr_diff_matrix)

            sum_vals = 0
            for template_i in range(0, template_height):
                for template_j in range(0, template_width):
                    template_val = template[template_i, template_j]
                    image_val = image[result_i + template_i, result_j + template_j]

                    template_val_float64 = np.float64(template_val)
                    image_val_float64 = np.float64(image_val)

                    diff_val = template_val_float64 - image_val_float64
                    sqr_diff_val = diff_val * diff_val
                    sum_vals += sqr_diff_val
            current_val_float_1_64 = sum_vals

            current_val_float_32 = np.float32(current_val_float_64)
            result_float_32[result_i, result_j] = current_val_float_32

            reference_val_float_32 = reference_matchTemplate[result_i, result_j]

            c = np.float64(661466430) - np.float64(661466400)

            diff_from_reference = current_val_float_32 - reference_val_float_32
            sqr_diff_from_reference = diff_from_reference * diff_from_reference
            david = 5


    return result_float_32


def get_temporary_image_and_template():
    image_height = 6
    image_width = 9

    image_1D = np.arange(start=1, stop=image_height * image_width + 1, step=1, dtype=np.uint8)
    image_2d = np.reshape(image_1D, [image_height, image_width])
    #image = cv2.merge((image_2d, image_2d, image_2d))
    image = image_2d

    template_height = 4
    template_width = 5

    template_1D = np.arange(start=1, stop=template_height * template_width + 1, step=1, dtype=np.uint8)
    template_2d = np.reshape(template_1D, [template_height, template_width])
    #template = cv2.merge((template_2d, template_2d, template_2d))
    template = template_2d

    template_uint8 = template
    image_uint8 = image
    return image_uint8, template_uint8

def get_image_with_identified_templates(rgb_image, template_images):
    rgb_image_with_identified_templates = rgb_image.copy()
    template_images = [template_images[0]]
    for rgba_template in template_images:
        template_height = rgba_template.shape[0]
        template_width = rgba_template.shape[1]
        blue_channel, green_channel, red_channel, alpha_channel = cv2.split(rgba_template)
        alpha_channel_only_0_or_255 = alpha_channel.copy()
        alpha_channel_only_0_or_255[alpha_channel > 0] = 255

        alpha_channel_only_0_or_1 = alpha_channel_only_0_or_255.copy()
        alpha_channel_only_0_or_1[alpha_channel_only_0_or_1 == 255] = 1
        mask = alpha_channel > 0



        grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        rgb_template = cv2.cvtColor(rgba_template, cv2.COLOR_RGBA2RGB)
        grayscale_template = cv2.cvtColor(rgb_template, cv2.COLOR_BGR2GRAY)



        # res1 = cv2.matchTemplate(grayscale_image, grayscale_template, cv2.TM_CCOEFF_NORMED, alpha_channel_only_0_or_255)
        # res2 = cv2.matchTemplate(
        #                         image=grayscale_image,
        #                         templ=grayscale_template,
        #                         method=cv2.TM_CCOEFF_NORMED,
        #                         mask=alpha_channel_only_0_or_255
        #                         )
        res3 = cv2.matchTemplate(image=grayscale_image, templ=grayscale_template, method=cv2.TM_SQDIFF_NORMED, mask=alpha_channel_only_0_or_1)

        #grayscale_image, grayscale_template = get_temporary_image_and_template()

        #result = cv2.matchTemplate(grayscale_image, grayscale_template, cv2.TM_SQDIFF)
        #result_david = calc_TM_SQDIFF(grayscale_image, grayscale_template)

        result_matchTemplate = cv2.matchTemplate(grayscale_image, grayscale_template, cv2.TM_SQDIFF_NORMED)
        result_david = calc_TM_SQDIFF_NORMED(grayscale_image, grayscale_template)

        # diff_matrices = result - result_david
        # sqr_diff_matrices = diff_matrices * diff_matrices
        # sum_sqr_diff_matrices = np.sum(sqr_diff_matrices)
        # are_equal = sum_sqr_diff_matrices < 1
        # if are_equal:
        #     print("Same matrices!")
        # else:
        #     print("Not same! something is wrong!")

        # Define a threshold
        threshold = 0.01
        tuple_template_start_points = np.where(result_david <= threshold)
        template_start_points = np.column_stack(tuple_template_start_points)

        filtered_points = calc_filter_points(points=template_start_points, threshold=10)


        num_of_matches = filtered_points.shape[0]

        rectangle_color = (0, 0, 255)
        rectangle_thickness = 4
        for i in range(num_of_matches):
            current_point = filtered_points[i]
            flipped_current_point = np.flip(current_point)
            int_flipped_current_point = flipped_current_point.astype(int)
            point1 = (int_flipped_current_point[0], int_flipped_current_point[1])
            point2 = (int_flipped_current_point[0] + template_width, int_flipped_current_point[1] + template_height)
            cv2.rectangle(img=rgb_image_with_identified_templates,
                          pt1=point1,
                          pt2=point2,
                          color=rectangle_color,
                          thickness=rectangle_thickness
                          )

        bgr_image_with_identified_templates = cv2.cvtColor(rgb_image_with_identified_templates, cv2.COLOR_RGB2BGR)
        plt.subplot(131), plt.imshow(result_matchTemplate, cmap='gray')
        plt.title('result_matchTemplate'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(result_david, cmap='gray')
        plt.title('result_david'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(bgr_image_with_identified_templates)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle('cv2.TM_CCOEFF_NORMED')

        plt.show()
    return rgb_image_with_identified_templates

def create_template_with_new_mask(rgba_template):
    blue_channel, green_channel, red_channel, alpha_channel = cv2.split(rgba_template)
    new_alpha_channel = np.zeros_like(alpha_channel)
    template_height = rgba_template.shape[0]
    template_width = rgba_template.shape[1]
    start_x = 50
    start_y = 50

    end_x = template_width - start_x
    end_y = template_height - start_y
    cv2.rectangle(new_alpha_channel, (start_x, start_y), (end_x, end_y), 255, -1)
    new_rgba_template = cv2.merge((blue_channel, green_channel, red_channel, new_alpha_channel))
    return new_rgba_template


def main():
    current_folder_full_path = os.getcwd()
    image_name = 'puppy.jpg'
    template_name = 'smiley160px.png'
    template_output_tiff_name = 'smiley160px.tiff'
    image_full_path = os.path.join(current_folder_full_path, image_name)
    template_full_path = os.path.join(current_folder_full_path, template_name)
    template_output_tiff_full_path = os.path.join(current_folder_full_path, template_output_tiff_name)
    start_points = np.array([[1500, 1000], [200, 400], [800, 900]])



    rgb_image = cv2.imread(image_full_path, cv2.IMREAD_UNCHANGED)
    orig_rgba_template = cv2.imread(template_full_path, cv2.IMREAD_UNCHANGED)
    rgba_template = orig_rgba_template
    #rgba_template = create_template_with_new_mask(orig_rgba_template)
    rgb_image_with_template = create_image_with_templates(rgb_image, rgba_template, start_points)

    #cv2.imwrite(template_output_tiff_full_path, rgba_template)

    # image_name = 'iron_dome.jpeg'
    # image_full_path = os.path.join(current_folder_full_path, image_name)
    # rgb_image_with_template = cv2.imread(image_full_path, cv2.IMREAD_UNCHANGED)

    template_images = get_all_template_images(rgba_template)

    #sumsqdiffr(rgb_image_with_template, rgba_template)
    #result_TM_SQDIFF = calc_TM_SQDIFF(rgb_image_with_template, rgba_template)
    #res3 = cv2.matchTemplate(grayscale_image, grayscale_template, cv2.TM_SQDIFF)

    rgb_image_with_identified_templates = get_image_with_identified_templates(rgb_image_with_template, template_images)

    resized_rgb_image_with_identified_templates = resize_image(image=rgb_image_with_identified_templates, scale_percent=70)
    cv2.imshow('Detecting smileys', resized_rgb_image_with_identified_templates)
    cv2.waitKey(0)



# def func2():
#     current_folder_full_path = os.getcwd()
#     image_name = 'puppy.jpg'
#     template_name = 'smiley160px.png'
#     image_full_path = os.path.join(current_folder_full_path, image_name)
#     template_full_path = os.path.join(current_folder_full_path, template_name)
#     start_points = np.array([[1500, 1000], [200, 400], [800, 900]])
#
#     rgb_image_with_template = create_image_with_templates(image_full_path, template_full_path, start_points)
#
#     # image_name = 'iron_dome.jpeg'
#     # image_full_path = os.path.join(current_folder_full_path, image_name)
#     # rgb_image_with_template = cv2.imread(image_full_path, cv2.IMREAD_UNCHANGED)
#
#     img = rgb_image_with_template
#     #img[:, :, :2] = 255  # change colors to red
#
#     rgba_template = cv2.imread(template_full_path, cv2.IMREAD_UNCHANGED)
#     template_all = rgba_template
#     template = template_all[:, :, 0:3]
#     template[:, :, 2] = 255  # change colors to red
#
#     alpha = template_all[:, :, 3]
#     alpha = cv2.merge([alpha, alpha, alpha])
#
#     h, w = template.shape[:2]
#
#     #method = cv2.TM_CCORR_NORMED
#     method = cv2.TM_CCOEFF_NORMED
#     res = cv2.matchTemplate(img, template, method, mask=alpha)
#
#     loc = np.where(res >= 0.8)
#     result = img.copy()
#
#     boxes = list()
#     for pt in zip(*loc[::-1]):
#         boxes.append((pt[0], pt[1], pt[0] + w, pt[1] + h))
#
#     boxes = imutils.object_detection.non_max_suppression(np.array(boxes))
#
#     for (x1, y1, x2, y2) in boxes:
#         cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 5)
#
#     resized_result = resize_image(image=result, scale_percent=50)
#     cv2.imshow('resized_result', resized_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    #func2()
    main()
