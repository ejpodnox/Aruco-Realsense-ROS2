# Individual ArUco marker detection with diagnostic information

int64 marker_id                    # Unique identifier of the marker
geometry_msgs/Pose pose            # Position and orientation of the marker
float32 confidence                 # Detection confidence (0.0 to 1.0)
float32 size_in_image              # Size of the marker in the image (pixels)
geometry_msgs/Point centroid_2d    # 2D centroid of the marker in image coordinates
bool is_valid                      # Whether this marker pose estimation is valid
int32 num_corners_detected         # Number of corners detected for this marker
