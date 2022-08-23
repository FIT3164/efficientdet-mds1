import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

class Image:
	def __init__(self, id, file_name, width=512, height=512, date_captured="2022/8/4", license=1, coco_url="", flickr_url=""):
		self.id = id
		self.file_name = file_name
		self.width = width
		self.height = height
		self.date_captured = date_captured
		self.license = license
		self.coco_url = coco_url
		self.flickr_url = flickr_url

	def to_dict(self):
		return {
			"id": self.id,
			"file_name": self.file_name,
			"width": self.width,
			"height": self.height,
			"date_captured": self.date_captured,
			"license": self.license,
			"coco_url": self.coco_url,
			"flickr_url": self.flickr_url
		}

class Annotation:
	def __init__(self, id, image_id, category_id, xmin, ymin, xmax, ymax, iscrowd=0):
		self.id = id
		self.image_id = image_id
		self.category_id = category_id
		width, height = xmax-xmin, ymax-ymin
		self.area = width * height
		self.bbox = [xmin, ymin, width, height]
		self.segmentation = [
			xmin,
			ymin,
			xmin+width,
			ymin,
			xmin+width,
			ymin+height,
			xmin,
			ymin+height
		]
		self.iscrowd = iscrowd

	def to_dict(self):
		return {
			"id": self.id,
			"image_id": self.image_id,
			"category_id": self.category_id,
			"iscrowd": self.iscrowd,
			"area": int(self.area),
			"bbox": [int(x) for x in self.bbox],
			"segmentation": [int(x) for x in self.segmentation]
		}

	def from_dict(d):
		[xmin, ymin, width, height] =  d["bbox"]
		return Annotation(d["id"], d["image_id"], d["category_id"], xmin, ymin, xmin+width, ymin+height)

	def topLeft(self):
		return (self.bbox[0], self.bbox[1])

	def topRight(self):
		return (self.segmentation[2], self.segmentation[3])

	def bottomRight(self):
		return (self.bbox[0]+self.bbox[2], self.bbox[1]+self.bbox[3])

	def bottomLeft(self):
		return (self.segmentation[6], self.segmentation[7])

	def end2end(self):
		return self.topLeft(), self.bottomRight()

	def shape(self):
		return self.bbox[2], self.bbox[3]

def plot_bbox(img, bbox):
	cpy = img.copy()
	[xmin, ymin, width, height] = bbox
	cv2.circle(cpy, (xmin, ymin), 5, (0,255,0), -1)
	cv2.circle(cpy, (xmin+width, ymin+height), 5, (0,255,0), -1)
	cv2.rectangle(cpy,(xmin,ymin),(xmin+width, ymin+height),(255,0,0),2)
	plt.imshow(cpy)

def pad_image(image, height, width, w=512, h=512):
    image = cv2.copyMakeBorder(
        image,
        (h-height)//2, (h-height)//2, (w-width)//2, (w-width)//2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return image, (h-height)//2

def crop_to_center(old_img, new_img):
	if isinstance(old_img, tuple):
		original_shape = old_img
	else:
		original_shape = old_img.shape
	original_width = original_shape[1]
	original_height = original_shape[0]
	original_center_x = original_shape[1] / 2
	original_center_y = original_shape[0] / 2
	new_width = new_img.shape[1]
	new_height = new_img.shape[0]
	new_center_x = new_img.shape[1] / 2
	new_center_y = new_img.shape[0] / 2
	new_left_x = int(max(new_center_x - original_width / 2, 0))
	new_right_x = int(min(new_center_x + original_width / 2, new_width))
	new_top_y = int(max(new_center_y - original_height / 2, 0))
	new_bottom_y = int(min(new_center_y + original_height / 2, new_height))
	canvas = np.zeros(original_shape)
	left_x = int(max(original_center_x - new_width / 2, 0))
	right_x = int(min(original_center_x + new_width / 2, original_width))
	top_y = int(max(original_center_y - new_height / 2, 0))
	bottom_y = int(min(original_center_y + new_height / 2, original_height))
	canvas[top_y:bottom_y, left_x:right_x] = new_img[new_top_y:new_bottom_y, new_left_x:new_right_x]
	return canvas.astype(np.uint8)

def flip_point_aux(dim, point, is_horizontal):
	(h,w) = dim
	px, py = point
	if is_horizontal:
		qx = w-px
		qy = py
	else:
		qx = px
		qy = h-py
	return int(round(qx)), int(round(qy))

def flip_img(img, is_horizontal=True):
	is_horizontal = 1 if is_horizontal else 0
	image = cv2.flip(img, is_horizontal)
	return image

def flip_point(annotation, new_img_id, new_annot_id, is_horizontal=1, w=512, h=512):
	(cX, cY) = (w // 2, h // 2)

	c1 = annotation.topLeft()
	c2 = annotation.topRight()
	c3 = annotation.bottomRight()
	c4 = annotation.bottomLeft()

	c1 = flip_point_aux((h, w), c1, is_horizontal)
	c2 = flip_point_aux((h, w), c2, is_horizontal)
	c3 = flip_point_aux((h, w), c3, is_horizontal)
	c4 = flip_point_aux((h, w), c4, is_horizontal)

	x_coords, y_coords = zip(c1, c2, c3, c4)
	xmin, ymin = min(x_coords), min(y_coords)
	xmax, ymax = max(x_coords), max(y_coords)

	# cv2.rectangle(rotated_img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
	new_annot = Annotation(new_annot_id, new_img_id, annotation.to_dict()["category_id"], xmin, ymin, xmax, ymax)
	return new_annot

def rotate_point_aux(origin, point, angle):
	angle = -angle * math.pi/180
	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return int(round(qx)), int(round(qy))

def rotate_img(img, angle=45):
	(h, w) = img.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	rotated_img = cv2.warpAffine(img, M, (nW, nH))
	rotated_img = crop_to_center(img, rotated_img)

	return rotated_img

def rotate_points(annotation, new_img_id, new_annot_id, angle, w=512, h=512):
	(cX, cY) = (w // 2, h // 2)
	c1 = annotation.topLeft()
	c2 = annotation.topRight()
	c3 = annotation.bottomRight()
	c4 = annotation.bottomLeft()

	c1 = rotate_point_aux((cX, cY), c1, angle)
	c2 = rotate_point_aux((cX, cY), c2, angle)
	c3 = rotate_point_aux((cX, cY), c3, angle)
	c4 = rotate_point_aux((cX, cY), c4, angle)

	x_coords, y_coords = zip(c1, c2, c3, c4)
	xmin, ymin = min(x_coords), min(y_coords)
	xmax, ymax = max(x_coords), max(y_coords)

	# cv2.rectangle(rotated_img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
	new_annot = Annotation(new_annot_id, new_img_id, annotation.to_dict()["category_id"], xmin, ymin, xmax, ymax)
	return new_annot

def init_dict():
    d = dict()
    d["info"] = {
        "description": "",
        "url": "",
        "version": "",
        "year": 2020,
        "contributor": "",
        "data_created": "2022-8-3"
    },
    d["licenses"] = {
        "id": 1,
        "name": None,
        "url": None
    }
    d["categories"] = [
        {
            "id": 1,
            "name": "good_chili",
            "supercategory": "None"
        },
        {
            "id": 2,
            "name": "bad_chili",
            "supercategory": "None"
        }
    ]
    d["images"] = []
    d["annotations"] = []
    return d