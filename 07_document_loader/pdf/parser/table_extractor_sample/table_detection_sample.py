import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

import pytesseract
from pytesseract import Output

from ultralyticsplus import YOLO, render_result
from PIL import Image

image = "./sample/img.png"

img = Image.open(image)
print(img)

# load model
model = YOLO("keremberke/yolov8m-table-extraction")

# set model parameters
model.overrides["conf"] = 0.25  # NMS confidence threshold
model.overrides["iou"] = 0.45  # NMS IoU threshold
model.overrides["agnostic_nms"] = False  # NMS class-agnostic
model.overrides["max_det"] = 1000  # maximum number of detections per image

# perform inference
results = model.predict(img)

# observe results
print("Boxes: ", results[0].boxes)
render = render_result(model=model, image=img, result=results[0])
print(render)

x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.numpy()[0])
img = np.array(Image.open(image))
# cropping
cropped_image = img[y1:y2, x1:x2]
cropped_image = Image.fromarray(cropped_image)
print(cropped_image)

ext_df = pytesseract.image_to_data(
    cropped_image, output_type=Output.DATAFRAME, config="--psm 6 --oem 3"
)
ext_df.head()
