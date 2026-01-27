import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/billy-linux-think/AAAA/ArucoRotation/src/install/aruco_z_rotation'
