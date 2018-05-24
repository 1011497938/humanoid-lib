bool save_image
bool kick
bool goalie_attacking

uint8 current_role

float32 time_to_reach_ball
float32 time_to_reach_defender
float32 time_to_reach_striker
float32 time_to_reach_mid_fielder

geometry_msgs/Vector3 dest
geometry_msgs/Vector3 final_dest

uint8 Striker = 1
uint8 Defender = 2
uint8 Goalie = 3
uint8 MidFielder = 4
