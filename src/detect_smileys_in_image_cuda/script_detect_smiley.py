import cv2
import numpy as np
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


def calc_filter_points(points, threshold):
    # Filter points so that the returned array will not have two points with distance<threshold between them
    # I assume that close points correspond to the same smiley
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

def match_template_sqdiff_normed(image, template, template_index, num_of_templates, mask=None):
    matrix_type = np.float32
    image = image.astype(matrix_type)
    template = template.astype(matrix_type)

    image_height = image.shape[0]
    image_width = image.shape[1]
    template_height = template.shape[0]
    template_width = template.shape[1]
    result_height = image_height - template_height + 1
    result_width = image_width - template_width + 1
    result = np.zeros((result_height, result_width), dtype=matrix_type)

    eps = 1e-8

    for i in range(result_height):
        print(f'{template_index + 1} out of {num_of_templates},  {i + 1} out of {result_height}')
        for j in range(result_width):
            patch = image[i:i + template_height, j:j + template_width]
            if mask is not None:
                diff = (patch - template) * (mask / 255)
            else:
                diff = patch - template
            nominator = np.sum(diff ** 2)

            sqr_template = template ** 2
            sum_sqr_template = np.sum(sqr_template)
            sqr_patch = patch ** 2
            sum_sqr_patch = np.sum(sqr_patch)
            sqr_denominator = sum_sqr_template * sum_sqr_patch
            denominator = np.sqrt(sqr_denominator)

            current_val = 1  # worst value in TM_SQDIFF_NORMED
            if denominator > eps:
                current_val = nominator / denominator
            result[i, j] = current_val
    return result

def get_image_with_identified_templates(rgb_image, template_images):
    rgb_image_with_identified_templates = rgb_image.copy()
    #template_images = [template_images[9]]
    num_of_templates = len(template_images)
    for template_index, rgba_template in enumerate(template_images):
        template_height = rgba_template.shape[0]
        template_width = rgba_template.shape[1]
        _, _, _, alpha_channel = cv2.split(rgba_template)
        alpha_channel_only_0_or_255 = alpha_channel.copy()
        alpha_channel_only_0_or_255[alpha_channel > 0] = 255
        grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        rgb_template = cv2.cvtColor(rgba_template, cv2.COLOR_RGBA2RGB)
        grayscale_template = cv2.cvtColor(rgb_template, cv2.COLOR_BGR2GRAY)

        # result = cv2.matchTemplate(image=grayscale_image, templ=grayscale_template, method=cv2.TM_SQDIFF_NORMED, mask=alpha_channel_only_0_or_255)
        result = match_template_sqdiff_normed(image=grayscale_image, template=grayscale_template, template_index=template_index, num_of_templates=num_of_templates, mask=alpha_channel_only_0_or_255)
        threshold = 0.02

        tuple_template_start_points = np.where(result <= threshold)
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
    return rgb_image_with_identified_templates

def create_image_with_templates(rgb_image, orig_rgba_template, start_points):
    template_images = get_all_template_images(orig_rgba_template)

    rgba_template = template_images[2]
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
            rgb_patch[:, :, current_channel] = patch_with_template_current_channel
            david = 4
        rgb_image_with_template[start_row:end_row, start_column:end_column, :] = rgb_patch
    #bgr_image_with_template = cv2.cvtColor(rgb_image_with_template, cv2.COLOR_RGB2BGR)

    return rgb_image_with_template

def main():
    current_folder_full_path = os.getcwd()
    image_name = 'iron_dome.jpeg'
    template_name = 'smiley160px.png'
    image_full_path = os.path.join(current_folder_full_path, image_name)
    template_full_path = os.path.join(current_folder_full_path, template_name)
    rgb_image_with_templates = cv2.imread(image_full_path, cv2.IMREAD_UNCHANGED)
    rgba_template = cv2.imread(template_full_path, cv2.IMREAD_UNCHANGED)
    template_images = get_all_template_images(rgba_template)

    # image_name = 'puppy.jpg'
    # image_full_path = os.path.join(current_folder_full_path, image_name)
    # rgb_image = cv2.imread(image_full_path, cv2.IMREAD_UNCHANGED)
    # start_points = np.array([[1500, 1000], [200, 400], [800, 900]])
    # rgb_image_with_templates = create_image_with_templates(rgb_image, rgba_template, start_points)


    rgb_image_with_identified_templates = get_image_with_identified_templates(rgb_image_with_templates, template_images)
    resized_rgb_image_with_identified_templates = resize_image(image=rgb_image_with_identified_templates, scale_percent=70)
    cv2.imshow('Detecting smileys', resized_rgb_image_with_identified_templates)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
