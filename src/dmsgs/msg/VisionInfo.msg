bool see_field
bool see_line
bool see_circle
bool see_goal # todo remove this, if goals.size() == 0, then we didn't see goal. if goals.size() == 1, we saw unknown goal, if goal.size() == 2, goals[0] is left goal, ..
bool see_ball
bool see_obstacle
bool loc_ok

# TODO(MWX): use vector for all, and remove bool values as much as possible
geometry_msgs/Vector3 robot_pos
geometry_msgs/Vector3 ball_field
geometry_msgs/Vector3 ball_global
geometry_msgs/Vector3 ball_velocity

geometry_msgs/Vector3 circle_field
geometry_msgs/Vector3 circle_global

geometry_msgs/Vector3[] goals_field
geometry_msgs/Vector3[] goals_global

geometry_msgs/Vector3[] obstacles_field
geometry_msgs/Vector3[] obstacles_global

Line[] lines_field
geometry_msgs/Vector3[] viewRange
ParticleMsg[] particles

# TODO(MWX): just use two different instance of VisionInfo ,remove simBlabla...
# Messages from simulator
float32 simYaw
geometry_msgs/Vector3 simFieldCircleCenter
geometry_msgs/Vector3[] simFieldWhitePoints
geometry_msgs/Vector3[] locFieldWhitePoints

HeadCommand ballTrack

HeadCommand circleTrack
HeadCommand redGoalTrack # -450, 0
HeadCommand blueGoalTrack # 450, 0

BehaviorInfo behaviorInfo
