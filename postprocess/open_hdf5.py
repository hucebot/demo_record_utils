import cv2
import h5py
import numpy as np

filename = "<filename>.h5"

with h5py.File(filename, "r") as f:

    key = "000/observations/images/cam_head_depth"

    # print some info
    print("encoded_img.dtype", f[key][0].dtype)
    print("encoded_img.shape", f[key][0].shape)
    
    num_frames = len(f[key])
    idxs = np.linspace(0, num_frames - 1, 10, dtype=int).tolist()
    for i in idxs:
        image = cv2.imdecode(f[key][i], cv2.IMREAD_UNCHANGED)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
