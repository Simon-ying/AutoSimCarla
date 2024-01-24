# std_msgs/Header.msg
# uint32 seq
# time stamp
# string frame_id
Header header 

int32[] startRingIndex
int32[] endRingIndex

float32 startOrientation
float32 endOrientation
float32 orientationDiff

bool[]    segmentedCloudGroundFlag # true - ground point, false - other points
uint32[]  segmentedCloudColInd # point column index in range image
float32[] segmentedCloudRange # point range 