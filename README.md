# 439GradSports
Final Project for 6.439

The /data folder contains a sample of 1 game's player movement data and event gamelog (12-25-2015 Clippers @ Lakers)

TODOS:
1) Make features file from game data
    -choose specific features (number of seconds since receiving ball, etc.)
    -examine variables that may be correlated to one another
2) Models
    -Logistic regression
    -Player clustering

Investigate the speed vs miss/make

assumptions being made:
-our method of finding the speed at time of shot takes a look at the max of the z position of the ball and then reverting back to the previous local min in z position of ball to find the frame at which the shooter begins his wind up. Our assumption here is that the velocity of the shooter does not change between time of windup and time of release. 
