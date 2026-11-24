# Slot features found in one camera image (pixels).
std_msgs/Header header
uint16 image_width
uint16 image_height

bool back_found
float32 back_y
float32 back_slope

bool left_found
bool right_found
float32 left_x
float32 right_x
float32 center_x

bool vp_found
vision_msgs/Point2D vp