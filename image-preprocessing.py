import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageOps
import numpy as np
import cv2


def remove_borders(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    diff = image_erosion(diff, 1)
    bbox = diff.getbbox(alpha_only=False)
    if bbox:
        img = img.crop(bbox)
    else:
        print("empty image")
    return bbox, diff


def image_dilataion(img, dilatation_size=1):
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(
        dilation_shape,
        (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        (dilatation_size, dilatation_size),
    )
    return convert_from_cv2_to_image(cv2.dilate(np.array(img), element))


def image_erosion(img, erosion_size=1):
    erosion_shape = cv2.MORPH_CROSS
    element = cv2.getStructuringElement(
        erosion_shape,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )
    return convert_from_cv2_to_image(cv2.erode(np.array(img), element))


# https://stackoverflow.com/a/65634189
def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# https://stackoverflow.com/a/65634189
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
def simple_thresholding(pil_img):
    open_cv_image = convert_from_image_to_cv2(pil_img)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # good but it removes too much
    # 150 is too much, lines are merged
    # 130 is still too much, some lines are merged
    # 120 is good, but too much information is lost
    ret, thresh1 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

    return convert_from_cv2_to_image(thresh1)


# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-2-adaptive-thresholding/
def adaptive_thresholding(pil_img):
    open_cv_image = convert_from_image_to_cv2(pil_img)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    thresh1 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5
    )

    # Creates too many white dots inside the fingerprint
    thresh2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5
    )

    return convert_from_cv2_to_image(thresh1), convert_from_cv2_to_image(thresh2)


# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-3-otsu-thresholding/
# https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
def otsu_thresholding(pil_img):
    open_cv_image = convert_from_image_to_cv2(pil_img)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret2, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return (
        convert_from_cv2_to_image(blur),
        convert_from_cv2_to_image(thresh1),
        convert_from_cv2_to_image(thresh2),
    )


def add_border(img, border_size=3):
    return ImageOps.expand(img, border=border_size, fill="black")


def process_image(img_path):
    print(img_path)
    img = Image.open(img_path).resize(img_dim, Image.LANCZOS)
    original_img = img
    dilation_img = image_dilataion(img)
    img = dilation_img
    simple_thresholding_img = simple_thresholding(img)
    (
        adaptive_thresholding_mean_img,
        adaptive_thresholding_gaussian_img,
    ) = adaptive_thresholding(img)
    (
        gaussian_filtered_img,
        otsu_thresholding_img,
        otsu_thresholding_gaussian_filtering_img,
    ) = otsu_thresholding(img)

    bbox, diff = remove_borders(adaptive_thresholding_gaussian_img)
    cropped_img = adaptive_thresholding_gaussian_img.crop(bbox)
    cropped_resized_img = cropped_img.resize(img_dim, Image.LANCZOS)

    plot_data = {}
    plot_data["original"] = [original_img, "Original"]
    plot_data["dilation"] = [dilation_img, "Dilation"]
    plot_data["cropped"] = [cropped_img, "Cropped adaptive \nthresholding gaussian"]
    plot_data["cropped_resized"] = [cropped_resized_img, "Cropped and resized"]
    plot_data["diff"] = [diff, "Diff for cropping"]
    plot_data["simple_thresholding"] = [simple_thresholding_img, "Simple thresholding"]
    plot_data["adaptive_thresholding_mean"] = [
        adaptive_thresholding_mean_img,
        "Adaptive \nthresholding mean",
    ]
    plot_data["adaptive_thresholding_gaussian"] = [
        adaptive_thresholding_gaussian_img,
        "Adaptive \nthresholding gaussian",
    ]
    plot_data["gaussian_filtered"] = [gaussian_filtered_img, "Gaussian filtered image"]
    plot_data["otsu_thresholding"] = [otsu_thresholding_img, "Otsu thresholding"]
    plot_data["otsu_thresholding_gaussian_filtering"] = [
        otsu_thresholding_gaussian_filtering_img,
        "Otsu thresholding \nand Gaussian filtering",
    ]

    for key in plot_data:
        plot_data[key][0] = add_border(plot_data[key][0])
    return plot_data


img_dim = (300, 300)
root = "FVC2002_dataset/"
file_list = sorted(glob.glob(root + "*" + "/" + "*" + "_1.tif"))
destination = "FVC2002_dataset_preprocessed/"

for img_path in file_list:
    plot_data = process_image(img_path)

    images_to_plot = [
        "original",
        "dilation",
        "simple_thresholding",
        "adaptive_thresholding_mean",
        "adaptive_thresholding_gaussian",
        "otsu_thresholding",
        "otsu_thresholding_gaussian_filtering",
        "diff",
        "cropped_resized",
        "cropped",
    ]
    plots = 10
    fig, arr = plt.subplots(1, plots)
    plt.gray()
    plt.rcParams.update({"font.size": 2})
    for i in range(plots):
        arr[i].imshow(plot_data[images_to_plot[i]][0])
        arr[i].set_title(plot_data[images_to_plot[i]][1], wrap=True)
        arr[i].axis("off")

    # plt.show()
    # break

    plt.savefig(
        destination + img_path.split("/")[-1].split(".")[0] + ".png",
        dpi=900,
        bbox_inches="tight",
    )
    print("Saving to " + destination + img_path.split("/")[-1].split(".")[0] + ".png")

    img = plot_data["original"][0]
    gaussian_filtered_img = plot_data["gaussian_filtered"][0]
    fig, arr = plt.subplots(2, 1)
    arr[0].hist(np.array(img).ravel(), 256)
    arr[0].set_title("Histogram of original")
    arr[1].hist(np.array(gaussian_filtered_img).ravel(), 256)
    arr[1].set_title("Histogram of Gaussian filtered image")

    plt.savefig(
        destination + img_path.split("/")[-1].split(".")[0] + "_histogram.png",
        dpi=900,
        bbox_inches="tight",
    )
    print(
        "Saving to "
        + destination
        + img_path.split("/")[-1].split(".")[0]
        + "_histogram.png"
    )
    plt.close()
