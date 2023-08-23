import numpy as np
import cv2



img1 = cv2.imread( "Image_1.jpg" )
img2 = cv2.imread( "Image_11.jpg" )
def quaternion_rotation_matrix(q0, q1, q2, q3):
   
    r00 = 2 * (q0 * q0 + q1 * q1) -1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) -1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) -1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix
K1=np.array([[2267.57,0,239.5],[0,2267.57,319.5],[0,0,1]],dtype=np.float32)
K1s=np.array([[2111.88,0,239.5],[0,2111.88,319.5],[0,0,1]],dtype=np.float32)
R1s=quaternion_rotation_matrix(0.349737,0.464982,0.528127,0.618512)
R1ss=quaternion_rotation_matrix(-0.268025994,-0.280882001,0.725898027,0.56774801)
R1n=np.array([[-0.00572002 ,-0.999525, 0.0302954 ],[0.962713, -0.0136984, -0.270179 ],[0.270466, 0.0276202 ,0.962333]],dtype=np.float32)
R1ns=np.array([[-0.0171232, 0.999105 ,-0.0386879 ],[-0.0452175, -0.0394278, -0.998199 ],[-0.99883, -0.0153431, 0.0458522 ]],dtype=np.float32)
Rx=R1s.T@R1n
Ry=R1ss.T@R1ns
X=Rx@Ry.T  





img1 = img1 // 2
img2 = img2 // 2

H = K1@R1s@X@np.linalg.inv( K1s )@R1ss.T
print(H)
#H = np.linalg.inv(H)
#print( R11 @ R22.T )
rst1 = cv2.warpPerspective( img1, np.eye(3),  (img1.shape[1]*2,img1.shape[0]*2 ))
rst2 = cv2.warpPerspective( img2, H,  (img1.shape[1]*2,img1.shape[0]*2 ))
rst = rst1 + rst2 
rst3=cv2.resize(rst,(1280,960))


cv2.imshow("wnd1", rst3)
cv2.waitKey(1)

T = np.array( [[1,0,img1.shape[1]//2 ],[0,1,img1.shape[0]//2],[0,0,1]], dtype = np.float32 )
rst1 = cv2.warpPerspective( img1, T @ np.eye(3) ,  (img1.shape[1]*2,img1.shape[0]*2 ))
rst2 = cv2.warpPerspective( img2, T @ H ,  (img1.shape[1]*2,img1.shape[0]*2 ))
rst = rst1 + rst2 
rst5 = cv2.resize(rst,(640,480))
cv2.imshow("wnd2", rst5)
cv2.waitKey(0)
