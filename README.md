# Calibrate stereo camera use opencv 

## 1. Use take_photo.py to take photos
- I use a USB stereo camera, this camera outputs a single combined image (stitched left and right views) each time, so it's necessary to manually split the photo into left and right images for separate saving.

## 2. Use stereo_calib.py to calibrate stereo camera
- Modify the directory path of the photos as needed, then run the **camera_calib.py** script. During corner detection for each image set, a window will pop up. Finally, the calibration results will be saved in two file formats, and a window showing the rectified results of a selected set of images will pop up.

## 3. Generate Checkbox picture
- [https://calib.io/pages/camera-calibration-pattern-generator](https://calib.io/pages/camera-calibration-pattern-generator)
