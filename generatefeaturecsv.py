#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:40:27 2018

@author: karanbhuwalka
"""

import csv

#from Utils import *

gameID = '0021500440'
outfile = 'Results/3ptfeatures.csv'

Data = readJson(gameID)

all3pts = get_all_3pt(Data)
for shot in all3pts:
    shooterID = shot['shooterID']
    eventID = shot['eventID']
    movement = get_movements(Data, eventID, shooterID)
    t_shot = get_shot_index(movement) # Time in frames from beginning of the event
    t_catch = get_catch_index(movement, shooterID) # Same

    if t_catch == t_shot:
        print('Event {} - Shooter {} caught ball at {} and shot at {}'.format(
			eventID, shooterID, t_catch, t_shot))
        continue

    is_cns = catch_and_shoot(movement)

    t_with_ball = (t_shot - t_catch) / 25

    v_with_ball = shooter_velocity_between_frames(movement, shooterID, t_catch, t_shot)

    v_before_shot = dict()
	# Find average velocity over n secs before shot for n = 0.5, 1, 1.5, ... 5]
    for n in [x / 2.0 for x in range(1, 11)]:
        v_before_shot[n] = None
        if n < t_with_ball:
            v_before_shot[n] = shooter_velocity_between_frames(movement, shooterID, int(t_shot - 25*n), t_shot)
		# print('{} - {}'.format(n, v_before_shot[n]))
        
    v_before_catch = dict()
    for n in [x / 2.0 for x in range(1, 11)]:
        v_before_catch[n] = None
        start_frame = int(t_catch - 25*n)
        if start_frame > 0:
            v_before_catch[n] = shooter_velocity_between_frames(
				movement, shooterID, start_frame, t_catch)
    row=list()
    row.append(shooterID)
    row.append(eventID)
    row.append(is_cns)
    row.append(3*movement[0])
    row.append(t_with_ball)
    row.append(v_with_ball)
    row.append(shooter_dist_at_time(movement,t_shot))
    row.append(ball_angle(movement, t_shot))
    row.append(shooter_move_angle(movement, shooterID, t_catch, t_shot))
    row.append(shooter_move_tobasket(movement,shooterID, t_catch, t_shot))
    row.append(closest_defender_dist(movement, t_shot)[0])
    row.append(closest_defender_dist(movement, t_shot)[1])
    row.append(closest_defender_velocity(movement,t_catch,t_shot))
    [row.append(v_before_shot[x]) for x in v_before_shot]
    [row.append(v_before_catch[x]) for x in v_before_catch]
    
    with open(outfile, 'a+', newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(row)