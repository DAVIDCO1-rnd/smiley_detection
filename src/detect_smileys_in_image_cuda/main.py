import cv2
import numpy as np

def find_single(nums):
    result = 0
    for num in nums:
        result ^= num
    return result


def find_two_singles(nums):
    xor_result = 0

    for num in nums:
        xor_result ^= num

    # find the rightmost set bit in xor_result
    rightmost_set_bit = xor_result & -xor_result

    unique1 = 0
    unique2 = 0
    for num in nums:
        if num & rightmost_set_bit:
            unique1 ^= num
        else:
            unique2 ^= num
    return unique1, unique2

def func1():
    array1 = [4, 2, 3, 2, 4, 11, 3]
    unique_number = find_single(array1)
    print("The unique number is: ", unique_number)

    array2 = [4, 4, 11, 32]
    unique1, unique2 = find_two_singles(array2)
    print(f"The two unique numbers are: {unique1} and {unique2}")

def func2():
    x = np.zeros(10, dtype=np.float64)
    y = x.strides
    print(y)

    n = 10
    a = np.arange(n)

    matrix_type = np.float64

    image_height = 9
    image_width = 11
    image = np.reshape(100*np.arange(image_height * image_width, dtype=matrix_type), [image_height, image_width])

    template_height = 2
    template_width = 3
    template = np.reshape(1*np.arange(template_height * template_width, dtype=matrix_type), [template_height, template_width])


    result_height = image_height - template_height + 1
    result_width = image_width - template_width + 1
    shape = (result_height, result_width, template_height, template_width)
    image_strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    multiple_patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=image_strides)

    template_strides = (0, 0, template.strides[0], template.strides[1])
    multiple_templates = np.lib.stride_tricks.as_strided(template, shape=shape, strides=template_strides)

    diff_matrix = multiple_templates - multiple_patches
    sqr_diff_matrix = np.multiply(diff_matrix, diff_matrix)
    nominators_matrix = np.sum(sqr_diff_matrix, axis=(2, 3))

    sqr_template = template * template
    sum_sqr_template = np.sum(sqr_template)

    sqr_multiple_patches = multiple_patches * multiple_patches
    sum_sqr_patches = np.sum(sqr_multiple_patches, axis=(2, 3))

    sqr_denominators_matrix = sum_sqr_template * sum_sqr_patches
    denominators_matrix = np.sqrt(sqr_denominators_matrix)

    result_matrix = nominators_matrix / denominators_matrix

    g00 = multiple_patches[0, 0]
    g01 = multiple_patches[0, 1]
    g_bottom_right = multiple_patches[multiple_patches.shape[0]-1, multiple_patches.shape[1]-1]

    t00 = multiple_templates[0, 0]
    t01 = multiple_templates[0, 1]
    t_bottom_right = multiple_templates[multiple_templates.shape[0]-1, multiple_templates.shape[1]-1]

    b = np.lib.stride_tricks.as_strided(a, (n//2, n//2), (8, 4))
    david = 5

def func3():
    # Define the shapes and strides based on the original arrays
    result_height, result_width, template_height, template_width = 1141, 1791, 160, 160
    image_shape = (result_height + template_height - 1, result_width + template_width - 1)
    template_shape = (template_height, template_width)

    # Generate sample data
    image = np.random.randint(0, 256, size=image_shape, dtype=np.uint8)
    template = np.random.randint(0, 256, size=template_shape, dtype=np.uint8)

    # Create strided views of the image and template
    image_strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    template_strides = (0, 0, template.strides[0], template.strides[1])

    multiple_patches = np.lib.stride_tricks.as_strided(image, shape=(
    result_height, result_width, template_height, template_width), strides=image_strides)
    multiple_templates = np.lib.stride_tricks.as_strided(template, shape=(
    result_height, result_width, template_height, template_width), strides=template_strides)

    # Create memory-mapped files for the result
    result_filename = 'result.dat'
    result_shape = (result_height, result_width, template_height, template_width)
    result = np.memmap(result_filename, dtype=np.int16, mode='w+', shape=result_shape)

    # Define a chunk size that fits into memory
    chunk_size = 100  # Adjust based on available memory

    # Perform chunk-wise subtraction and store in the memory-mapped result
    for i in range(0, result_height, chunk_size):
        i_end = min(i + chunk_size, result_height)
        np.subtract(multiple_templates[i:i_end], multiple_patches[i:i_end], out=result[i:i_end])

    # Flush changes to disk
    result.flush()

    print("Subtraction done using memory-mapped files.")


def match_template_sqdiff_normed(image, template, mask=None):
    img_h, img_w = image.shape
    tpl_h, tpl_w = template.shape
    result = np.zeros((img_h - tpl_h + 1, img_w - tpl_w + 1), dtype=np.float32)

    for y in range(result.shape[0]):
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

def func4():
    # Load the source image and the template image in grayscale
    grayscale_image = cv2.imread('path_to_grayscale_image.jpg', cv2.IMREAD_GRAYSCALE)
    grayscale_template = cv2.imread('path_to_grayscale_template.jpg', cv2.IMREAD_GRAYSCALE)

    # Create or load a mask (same size as the template)
    # For simplicity, let's create a mask with all ones (i.e., no part is ignored)
    mask = np.ones(grayscale_template.shape, dtype=np.uint8) * 255

    # Perform template matching with the mask using the TM_SQDIFF_NORMED method
    res = match_template_sqdiff_normed(grayscale_image, grayscale_template, mask)

    # Find the best match location
    # For TM_SQDIFF_NORMED, the minimum value gives the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # The best match top-left corner
    top_left = min_loc
    h, w = grayscale_template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a rectangle around the matched region
    cv2.rectangle(grayscale_image, top_left, bottom_right, (0, 255, 0), 2)

    # Save or display the result
    cv2.imwrite('matched_result.jpg', grayscale_image)
    cv2.imshow('Matched Result', grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    func4()

main()