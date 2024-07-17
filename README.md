# Content
- [Shortcuts Opencv](#Shortcuts-opencv)
- [Scripts description](#Scripts-description)

### Shortcuts Opencv
| Command | Description |
| --- | --- |
| `img = cv2.resize(img, (800, 600))` | resize image
| `gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` | gray image from BGR |
| `img = cv2.circle(img, center, radius=5, color=(0,0, 255), thickness=2)` | draw circle |
| `img = cv2.rectangle(img, rect_start, rect_end, color=(0,0, 255), thickness=2)` | draw rectangle |
| ` img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)` | rotate image counterclockwise |
|`  img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` | rotate image clockwise |

### Scripts description
| Name | Description |
| --- | --- |
| gammaCorrector.py | Improve the brightness of the image |
| CameraCalibrationCPP | camera calibration  in c++ |


