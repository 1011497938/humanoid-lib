bool connected
uint8 state
uint8 secondaryState
bool firstHalf
bool kickoff

bool ourIndirectFreeKick
bool ourDirectFreeKick
bool ourPenaltyKick

bool enemyIndirectFreeKick
bool enemyDirectFreeKick
bool enemyPenaltyKick

bool state2Ready
bool state2Freeze

uint16 secsRemaining
uint16 secondaryTime

uint8 secsTillUnpenalised
uint8 ourScore
uint8 enemyScore
bool teamCyan
bool penalised

# STATE

uint8 INITIAL            =  0
uint8 READY              =  1
uint8 SET                =  2
uint8 PLAYING            =  3
uint8 FINISHED           =  4

# STATE2
uint8 NORMAL            =  0
uint8 PENALTYSHOOT      =  1
uint8 OVERTIME          =  2
uint8 TIMEOUT           =  3
uint8 DIRECT_FREEKICK   =  4
uint8 INDIRECT_FREEKICK =  5
uint8 PENALTYKICK       =  6
