# Content
- [Shortcuts Opencv](#Shortcuts-opencv)
- [Labels format](#Labels-format)
- [Scripts description](#Scripts-description)
- [ONNX format] (#ONNX-format)

### Shortcuts Opencv
| Command | Description |
| --- | --- |
| `img = cv2.resize(img, (800, 600))` | resize image
| `gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` | gray image from BGR |
| `img = cv2.circle(img, center, radius=5, color=(0,0, 255), thickness=2)` | draw circle |
| `img = cv2.rectangle(img, rect_start, rect_end, color=(0,0, 255), thickness=2)` | draw rectangle |
| ` img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)` | rotate image counterclockwise |
|`  img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` | rotate image clockwise |

Add text to the image
```
sample = cv2.putText(img=sample, text=str(label), org=(int(box[0]), int(box[1])),
                                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.0,
                                 color=(125,246,55), thickness=1, lineType=cv2.LINE_AA)
```

### Labels format
| Format | Dataset | Description | Normalized (divide by height and width) |
| --- | --- | --- | --- |
| pascal_voc |  Pascal VOC dataset | [x_min, y_min, x_max, y_max] | No |
| albumentations | ---| [x_min, y_min, x_max, y_max] | Yes |
| coco | coco | [x_min, y_min, width, height] | No |
| yolo | --- | [x_center, y_center, width, height] | Yes | 



### Scripts description
| Name | Description |
| --- | --- |
| gammaCorrector.py | Improve the brightness of the image |
| CameraCalibrationCPP | camera calibration  in c++ |

### ONNX format
1 install onnx
```
pip install onnx
```
1- convert model to onnx
```
python3 deploy/ONNX/export_onnx.py --weights ../latest_trained_models/yolov6l.pt --img 1280 --batch 1 --simplify --iou-thres 0.45 --conf-thres 0.25 --device cpu
```
or 
```
python3 deploy/ONNX/export_onnx.py --weights ../latest_trained_models/yolov6l.pt --img 1280 --batch 1 --simplify --iou-thres 0.45 --conf-thres 0.25 --device 0
```

