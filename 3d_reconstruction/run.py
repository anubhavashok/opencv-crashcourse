import cv2
import numpy as np
import matplotlib.pyplot as plt


def displayImage(img):
    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Write point cloud to ply file
def write_ply(f, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    f.write(ply_header % dict(vert_num=len(verts)))
    np.savetxt(f, verts, '%f %f %f %d %d %d')

# Load images

# Convert images to grayscale

# Create stereo resolver
stereo = cv2.StereoSGBM(1, 16, 15)

# Compute disparity (stereo.compute)

# Refine disparity map (blur and filterSpeckles)

# Convert disparity to point cloud
height, width, depth = imgL.shape
focal_length = 3
Q = np.float32([[1, 0, 0, width / 2],
                [0, -1, 0, height / 2],
                [0, 0, focal_length, 0],
                [0, 0, 0, 1]])

# Convert 2D representation into 3D points

# Write point cloud to disk
