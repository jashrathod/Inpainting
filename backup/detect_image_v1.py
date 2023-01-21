import cv2
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

    cv2.imwrite("result_v1.jpg", img)

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

    cv2.imwrite("black_and_white_v1.jpg", img_bw)
    ## print(img_bw.shape)
    return img_bw


def produce_overlay(img_bw):
    # src = cv2.imread("black_and_white.jpg", 1)
    src = img_bw
    ## print(src.shape)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite("overlay_v1.png", dst)


def end_to_end(image_path, DIMINISH_CATEGORY, overlay_type):
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
    img_bw = cv2.rectangle(img, start_point, end_point, BLACK_COLOR, -1)
    
    coordinates = [obj[2] for obj in results]
    
    for obj in coordinates:
        start_point = obj[0][0], obj[0][1]
        end_point = obj[2][0], obj[2][1]
        if overlay_type == 'white':
            img_bw = cv2.rectangle(img_bw, start_point, end_point, WHITE_COLOR, -1)
        # elif overlay_type == 'blur':
        #     roi = image[y:y+h, x:x+w]
        #     roi = cv2.GaussianBlur(roi, (15, 15), 30)
        #     image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    # tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    # b, g, r = cv2.split(img)
    # rgba = [b, g, r, alpha]
    # dst = cv2.merge(rgba, 4)
    mask_path = "black_and_white.jpg"
    cv2.imwrite(mask_path, img_bw)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    flag = cv2.INPAINT_TELEA
    # flag = cv2.INPAINT_NS
    dst = cv2.inpaint(img, mask, 3, flag)

    # cv2.imshow('dst', dst)
    cv2.imwrite("result.jpg", dst)


def main():
    DIMINISH_CATEGORY = ['Mobile phone', 'Bottle']
    # image_path = "IMG_6075.JPG"
    image_path = "base_image.jpg"
    # image_path = "image.jpg"
    objects = localize_objects(image_path)
    filtered_objects = [obj for obj in objects if obj[0] in DIMINISH_CATEGORY]
    # filtered_objects = objects
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
    # image_path = "image.jpg"
    end_to_end(image_path, DIMINISH_CATEGORY, 'white')


# main()
main2()
