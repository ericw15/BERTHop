import pydicom

def dicom_to_png(path):
    ds = pydicom.dcmread(path)
    im = ds.pixel_array
    im = cv2.resize(im, (206, 206))
    return im