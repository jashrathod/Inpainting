import cv2
import numpy as np
from google.cloud import vision


def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

    results = []
    for object_ in objects:
        item = [object_.name, object_.score]
        vertices = []
        for vertex in object_.bounding_poly.normalized_vertices:
            vertices.append([vertex.x, vertex.y])
        item.append(vertices)
        results.append(item)

    return results


def bounding_box(image_path, objects):
    BBCOLOR = (255, 0, 0)
    img = cv2.imread(image_path)

    img_width = img.shape[0]
    img_height = img.shape[1]

    coordinates = [obj[2] for obj in objects]

    for i in range(len(coordinates)):
        for j in range(len(coordinates[i])):
            coordinates[i][j][0] = int(coordinates[i][j][0] * img_height)
            coordinates[i][j][1] = int(coordinates[i][j][1] * img_width)

    for obj in coordinates:
        start = obj[0][0], obj[0][1]
        end = obj[2][0], obj[2][1]
        img = cv2.rectangle(img, start, end, BBCOLOR, 3)

    cv2.imwrite("result.jpg", img)

    return img, coordinates


def generate_bw_overlay(img_bw, objects):
    BLACK_COLOR = (0, 0, 0)
    WHITE_COLOR = (255, 255, 255)

    start_point = 0, 0
    end_point = int(img_bw.shape[1]), int(img_bw.shape[0])
    img_bw = cv2.rectangle(img_bw, start_point, end_point, BLACK_COLOR, -1)

    coordinates = [obj[2] for obj in objects]

    for obj in coordinates:
        start_point = obj[0][0], obj[0][1]
        end_point = obj[2][0], obj[2][1]
        img_bw = cv2.rectangle(img_bw, start_point, end_point, WHITE_COLOR, -1)

    cv2.imwrite("black_and_white.jpg", img_bw)
    # print(img_bw.shape)
    return img_bw


def produce_overlay(img_bw):
    # src = cv2.imread("black_and_white.jpg", 1)
    src = img_bw
    # print(src.shape)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite("overlay.png", dst)


def end_to_end(image_path, DIMINISH_CATEGORY, overlay_type, inpaint_method):
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

    results = []
    for object_ in objects:
        item = [object_.name, object_.score]
        vertices = []
        for vertex in object_.bounding_poly.normalized_vertices:
            vertices.append([vertex.x, vertex.y])
        item.append(vertices)
        results.append(item)

    # with open('object_detection_log.txt', 'w') as f:
    #     for line in results:
    #         f.write(f"{line}\n")

    results = [obj for obj in results if obj[0] in DIMINISH_CATEGORY]

    img = cv2.imread(image_path)

    img_width = img.shape[0]
    img_height = img.shape[1]

    coordinates = [obj[2] for obj in results]

    for i in range(len(coordinates)):
        for j in range(len(coordinates[i])):
            coordinates[i][j][0] = int(coordinates[i][j][0] * img_height)
            coordinates[i][j][1] = int(coordinates[i][j][1] * img_width)

    BLACK_COLOR = (0, 0, 0)
    WHITE_COLOR = (255, 255, 255)

    start_point = 0, 0
    end_point = img_height, img_width
    img_bw = img
    img_bw = cv2.rectangle(img_bw, start_point, end_point, BLACK_COLOR, -1)

    coordinates = [obj[2] for obj in results]

    for obj in coordinates:
        start_point = obj[0][0], obj[0][1]
        end_point = obj[2][0], obj[2][1]
        if overlay_type == 'white':
            img_bw = cv2.rectangle(img_bw, start_point,
                                   end_point, WHITE_COLOR, -1)

    mask_path = "black_and_white.jpg"
    cv2.imwrite(mask_path, img_bw)
    img_base = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if inpaint_method == "opencv":
        flag = cv2.INPAINT_TELEA
        dst = cv2.inpaint(img_base, mask, 3, flag)
    elif inpaint_method == "stable_diffusion":
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting")
        prompt_sd = "3D render of flowers"
        im_sd = Image.open(image_path)
        mask_sd = Image.open(mask_path)
        image_sd = pipe(prompt=prompt_sd, image=im_sd,
                        mask_image=mask_sd, num_inference_steps=5).images[0]
        image_sd.save("result_sd.jpg")
        dst = cv2.imread("result_sd.jpg")

    start_point = 0, 0
    end_point = img_height, img_width
    img_blur = img
    img_blur = cv2.bitwise_and(dst, dst, mask=mask)

    tmp = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img_blur)
    rgba = [b, g, r, alpha]
    trsp = cv2.merge(rgba, 4)
    cv2.imwrite("transparent_blurred.png", trsp)

    return coordinates[0][0]


def main():
    DIMINISH_CATEGORY = ['Mobile phone', 'Bottle']
    # image_path = "IMG_6075.JPG"
    image_path = "base_image.jpg"
    # image_path = "test_image.jpeg"
    # image_path = "image.jpg"
    objects = localize_objects(image_path)
    filtered_objects = [obj for obj in objects if obj[0] in DIMINISH_CATEGORY]
    cv2_img = bounding_box(image_path, filtered_objects)
    cv2_img_bw = generate_bw_overlay(cv2_img[0], filtered_objects)
    produce_overlay(cv2_img_bw)
    # with open('object_detection_log.txt', 'w') as f:
    #     for line in objects:
    #         f.write(f"{line}\n")


def main2():
    DIMINISH_CATEGORY = ['Mobile phone', 'Bottle']
    # image_path = "IMG_6075.JPG"
    image_path = "base_image.jpg"
    # image_path = "test_image.jpeg"
    # image_path = "image.jpg"
    top_left = end_to_end(image_path, DIMINISH_CATEGORY,
                          'white', 'opencv')
    
    print(top_left)


# main()
main2()
