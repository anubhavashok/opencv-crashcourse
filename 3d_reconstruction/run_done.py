import cv2
import numpy as np
import matplotlib.pyplot as plt


def displayImage(img):
    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

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


imgL = cv2.imread('./data/img1.png')
imgR = cv2.imread('./data/img2.png')

imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

displayImage(np.hstack((imgL, imgR)))
displayImage(np.hstack((imgL_gray, imgR_gray)))

stereo = cv2.StereoSGBM(1, 16, 15)

# Compute disparity
disparity = stereo.compute(imgL_gray, imgR_gray)

# Refine disparity map
disparity = cv2.blur(disparity, (5, 5))
cv2.filterSpeckles(disparity, 0, 1, 128)
displayImage(disparity.astype(np.int8))

# Convert disparity to point cloud
height, width, depth = imgL.shape
focal_length = 3
Q = np.float32([[1, 0, 0, width / 2],
                [0, -1, 0, height / 2],
                [0, 0, focal_length, 0],
                [0, 0, 0, 1]])

# Convert 2D representation into 3D points
points = cv2.reprojectImageTo3D(disparity, Q)
print(points[0][0])

colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)


# Write point cloud to disk
f = open('output.ply', 'w')
write_ply(f, points, colors)
