import logging
import math
import os
import matplotlib.pyplot as plt

import cv2
import numpy as np
from imutils import auto_canny
from PIL import Image, ImageStat
import rlsa
from skimage.measure.entropy import shannon_entropy
from tqdm import tqdm

from helpers import frame_number_filename_mapping
from text_detection import get_text_bounding_boxes, load_east

logger = logging.getLogger(__name__)

OUTPUT_PATH_MODIFIER = "_figure_"


def area_of_overlapping_rectangles(a, b):

    dx = min(a[0], b[0]) - max(a[2], b[2])  # xmax, xmax, xmin, xmin
    dy = min(a[1], b[1]) - max(a[3], b[3])  # ymax, ymax, ymin, ymin
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def detect_color_image(image, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    """Detect if an image contains color, is black and white, or is grayscale.
    Returns:
        str: Either "grayscale", "color", "b&w" (black and white), or "unknown".
    """
    pil_img = Image.fromarray(image)
    bands = pil_img.getbands()
    if bands == ("R", "G", "B") or bands == ("R", "G", "B", "A"):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum(
                (pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2]
            )
        MSE = float(SSE) / (thumb_size * thumb_size)
        if MSE <= MSE_cutoff:
            return "grayscale"
        return "color"
    if len(bands) == 1:
        return "b&w"
    return "unknown"


def convert_coords_to_corners(box):
    x, y, w, h = box
    x_values = (x + w, x)
    y_values = (y + h, y)
    rectangle = (max(x_values), max(y_values), min(x_values), min(y_values))
    return rectangle


def area_of_corner_box(box):
    return (box[0] - box[2]) * (box[1] - box[3])



def detect_figures(
    image_path,
    output_path=None,
    east="frozen_east_text_detection.pb",
    text_area_overlap_threshold=0.32,  # 0.15
    figure_max_area_percentage=0.60,
    text_max_area_percentage=0.30,
    large_box_detection=True,
    do_color_check=True,
    do_text_check=True,
    entropy_check=2.5,
    do_remove_subfigures=True,
    do_rlsa=False,
):
   
    image = cv2.imread(image_path)

    image_height = image.shape[0]
    image_width = image.shape[1]
    image_area = image_height * image_width

    if not output_path:
        file_parse = os.path.splitext(str(image_path))
        filename = file_parse[0]
        ext = file_parse[1]
        start_output_path = filename + OUTPUT_PATH_MODIFIER

    if do_text_check:
        text_bounding_boxes = get_text_bounding_boxes(image, east)
        # Remove boxes that are too large
        text_bounding_boxes = [
            box
            for box in text_bounding_boxes
            if area_of_corner_box(box)  # area of box
            < text_max_area_percentage * image_area
        ]

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blurred = cv2.GaussianBlur(gray_thresh, (3, 3), 0)

    canny = auto_canny(blurred)

    canny_dilated_large = cv2.dilate(canny, np.ones((22, 22), dtype=np.uint8))
    canny_dilated_small = cv2.dilate(canny, np.ones((3, 3), dtype=np.uint8))


    if do_rlsa:
        x, y = canny.shape
        value = max(math.ceil(x / 70), math.ceil(y / 70)) + 20  # heuristic
        rlsa_result = ~rlsa.rlsa(~canny, True, True, value)  # rlsa application
        canny_dilated_large = rlsa_result
        # cv2.imwrite('rlsah.png', rlsa_result)

    contours_large = cv2.findContours(
        canny_dilated_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_large = (
        contours_large[0] if len(contours_large) == 2 else contours_large[1]
    )


    bounding_boxes_large = np.array(
        [cv2.boundingRect(contour) for contour in contours_large]
    )

    if large_box_detection:
        contours_small = cv2.findContours(
            canny_dilated_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_small = (
            contours_small[0] if len(contours_small) == 2 else contours_small[1]
        )


    max_area = int(figure_max_area_percentage * image_area)
    min_area = (image_height // 3) * (image_width // 6)
    min_area_small = min_area

    padding = image_height // 70

    figures = []
    all_figure_boxes = []
    output_paths = []

    if large_box_detection:
        # none_tested = True
        for contour in contours_small:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)

            # Figure has 4 corners and it is convex
            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
                and min_area_small < cv2.contourArea(approx) < max_area
            ):
                # none_tested = False
                # min_area_small = cv2.contourArea(approx)
                figure_contour = approx[:, 0]

                # if not none_tested:
                bounding_box = cv2.boundingRect(figure_contour)
                x, y, w, h = bounding_box
                figure = original[
                    y - padding : y + h + padding, x - padding : x + w + padding
                ]
                figures.append(figure)
                all_figure_boxes.append(convert_coords_to_corners(bounding_box))

    for box in bounding_boxes_large:
        x, y, w, h = box
        area = w * h
        aspect_ratio = w / h
        if min_area < area < max_area and 0.2 < aspect_ratio < 6:
            # Draw bounding box rectangle, crop using numpy slicing
            roi_rectangle = convert_coords_to_corners(box)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.imwrite("rect.png", image)
            if (
                y + h >= image_height
                or x + w >= image_width
                or y <= image_height
                or x <= image_width
            ):
                potential_figure = original[y : y + h, x : x + w]
            else:
                potential_figure = original[
                    y - padding : y + h + padding, x - padding : x + w + padding
                ]
            # cv2.imwrite("potential_figure.png", potential_figure)

            # Go to next figure if the `potential_figure` is empty
            if potential_figure.size == 0:
                continue

            text_overlap_under_threshold = True
            roi_is_color = True

            if do_text_check:
                total_area_overlapped = sum(
                    area_of_overlapping_rectangles(roi_rectangle, text_rectangle)
                    for text_rectangle in text_bounding_boxes
                )
                logger.debug("Total area overlapped by text: %i", total_area_overlapped)
                text_overlap_under_threshold = (
                    total_area_overlapped < text_area_overlap_threshold * area
                )

            if do_color_check:
                roi_is_color = detect_color_image(potential_figure) == "color"

            checks_passed = roi_is_color and text_overlap_under_threshold

            if checks_passed:
                figures.append(potential_figure)
                all_figure_boxes.append(roi_rectangle)

    if do_remove_subfigures:
        remove_idxs = []
        for idx, figure in enumerate(all_figure_boxes):
            for compare_idx, figure_to_compare in enumerate(
                all_figure_boxes[idx + 1 :]
            ):
                overlapping_area = area_of_overlapping_rectangles(
                    figure, figure_to_compare
                )
                if overlapping_area > 0:
                    figure_area = area_of_corner_box(figure)
                    figure_to_compare_area = area_of_corner_box(figure_to_compare)
                    if figure_area > figure_to_compare_area:
                        remove_idxs.append(compare_idx)
                    else:
                        remove_idxs.append(idx)

        figures = [
            figure for idx, figure in enumerate(figures) if idx not in remove_idxs
        ]

    for idx, figure in enumerate(figures):
        if entropy_check:
            # If `entropy_check` is a boolean, then set it to the default
            if type(entropy_check) is bool and entropy_check:
                entropy_check = 2.5
            try:
                gray = cv2.cvtColor(figure, cv2.COLOR_BGR2GRAY)
            except:
                continue
            high_entropy = shannon_entropy(gray) > entropy_check
            if not high_entropy:
                continue

        full_output_path = start_output_path + str(idx) + ext
        output_paths.append(full_output_path)
        cv2.imwrite(full_output_path, figure)

    logger.debug("Number of Figures Detected: %i", len(figures))
    return figures, output_paths


def all_in_folder(
    path,
    remove_original=False,
    east="frozen_east_text_detection.pb",
    do_text_check=True,
    **kwargs
):

    figure_paths = []
    images = os.listdir(path)
    images.sort()

    if do_text_check:
        east = load_east(east)

    for item in tqdm(images, total=len(images), desc="> Figure Detection: Progress"):
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path) and OUTPUT_PATH_MODIFIER not in str(
            current_path
        ):
            # Above checks that file exists and does not contain `OUTPUT_PATH_MODIFIER` because that would
            # indicate that the file has already been processed.
            _, output_paths = detect_figures(
                current_path, east=east, do_text_check=do_text_check, **kwargs
            )

            figure_paths.extend(output_paths)
            if remove_original:
                os.remove(current_path)
    logger.debug("> Figure Detection: Returning figure paths")
    return figure_paths


def add_figures_to_ssa(ssa, figures_path):
    # If the SSA contains frame numbers
    if ssa and "frame_number" in ssa[0].keys():
        mapping = frame_number_filename_mapping(figures_path)

        for idx, slide in enumerate(ssa):
            current_slide_idx = slide["frame_number"]
            try:
                ssa[idx]["figure_paths"] = mapping[current_slide_idx]
            except KeyError:  # Ignore frames that have no figures
                pass

    return ssa



# all_in_folder("delete/")
# detect_figures("delete/img_01054_noborder.jpg")
# detect_figures(r'C:\Users\Administrator\Desktop\Figure\final_frames\1.jpg', output_path=r'output_folder/', do_text_check=False)
detect_figures(r'C:\Users\Administrator\Desktop\Figure\final_frames\3.jpg', east="frozen_east_text_detection.pb")
