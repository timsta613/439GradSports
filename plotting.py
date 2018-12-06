import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import math
import pandas as pd
from scipy.spatial import distance_matrix

# Update of readJson_test() which matches movement data to the event data by event ID
# Pass in the game ID to match
def readJson(gameID):
    movementfile = 'data/movement/{}.json'.format(gameID)
    with open(movementfile) as f:
        movement = json.load(f)

    # Creates dictionary with keys as event IDs and values as movement data
    # for each event ID found in the movement data
    movementsByEventID = {int(event['eventId']): event for event in movement['events']}

    movementByEvents = dict()

    eventsfile = 'data/events/{}.csv'.format(gameID)
    with open(eventsfile) as f:
        reader = csv.reader(f)
        for row in reader:
            # First line of the file is headers, so int(row[1]) will be invalid
            try:
                eventID = int(row[1])
            except ValueError:
                # This line does not have a proper event ID so we don't need to record it
                continue
            if eventID in movementsByEventID.keys() and len(movementsByEventID[eventID]['moments']) > 0:
                movementByEvents[eventID] = dict()
                movementByEvents[eventID]['eventData'] = row
                # Only keeps the player movement "moments" data, as other data is extraneous
                movementByEvents[eventID]['movementData'] = movementsByEventID[eventID]['moments']

    return movementByEvents

def get_movements(Data, eventID, shooterID):
    shooter_x = []
    shooter_y = []
    ball_x = []
    ball_y = []
    ball_z = []
    num_events = len(Data)

    try:
        movement_data = Data[eventID]['movementData']
    except:
        raise KeyError('Movement data for the given event ID {} is not present'.format(eventID))

    # Take 5th element (list of player & ball locations) of 1st moment in movement data
    # and construct dictionary of Player IDs : index in that list of players

    shooter_row_num = None
    for moment in movement_data:
        playerIndices = {data[1]: index for index, data in enumerate(moment[5])}
        if shooterID in playerIndices:
            shooter_row_num = playerIndices[shooterID]

    if shooter_row_num is None:
        return None
    # playerIndices = {data[1]: index for index, data in enumerate(movement_data[0][5])}
    # #print(playerIndices)
    # shooter_row_num = playerIndices[shooterID]
    # #print(shooter_row_num)

    for index, moment in enumerate(movement_data):
        locations = moment[5]
        # Sometimes 11 locations are present which indicates ball movement is being tracked
        if len(locations) == 11:
            shooter_x.append(locations[shooter_row_num][2])
            shooter_y.append(locations[shooter_row_num][3])
            ball_x.append(locations[0][2])
            ball_y.append(locations[0][3])
            ball_z.append(locations[0][4])
        # If only 10 rows, then no ball data is present so copy from before
        # Ball is always index 0 so shooter index decreases by 1 when ball isn't present
        elif len(locations) == 10:
            shooter_x.append(locations[shooter_row_num-1][2])
            shooter_y.append(locations[shooter_row_num-1][3])
            ball_x.append(movement_data[index-1][5][0][2])
            ball_y.append(movement_data[index-1][5][0][3])
            ball_z.append(movement_data[index-1][5][0][4])
        # If fewer than 10 rows, then we have no idea what's going on. Carry everything forward
        else:
            shooter_x.append(shooter_x[-1])
            shooter_y.append(shooter_y[-1])
            ball_x.append(ball_x[-1])
            ball_y.append(ball_y[-1])
            ball_z.append(ball_z[-1])

    if Data[eventID]['eventData'][2] == '1':
        miss_make = 1
    else:
        miss_make = 0

    return [miss_make, shooter_x, shooter_y, ball_x, ball_y, ball_z]

def get_all_3pt(Data):
    all_3pt_data = []
    num_events = len(Data)
    for event in Data:
        if Data[event]['eventData'][7].find('3PT') != -1 or Data[event]['eventData'][9].find('3PT') != -1:
            #print(event)
            eventID = int(Data[event]['eventData'][1])
            #print(eventID)
            shooterID = int(Data[event]['eventData'][13])
            #print(shooterID)
            all_3pt_data.append({'shooterID': shooterID, 'eventID': eventID, 'movements': get_movements(Data, eventID, shooterID)})
    return all_3pt_data

# movement arg is output of get_movements(Data, eventID, shooterID)
def get_shot_index(movement):
    shooter_x = movement[1]
    shooter_y = movement[2]
    ball_x = movement[3]
    ball_y = movement[4]
    ball_z = movement[5]
    # index_highest = ball_z.index(max(ball_z)) # Finds location of highest point of ball during event

    k = 20
    if len(ball_z) < k:
        k = len(ball_z)

    temp = ball_z[:k]
    indices = list(range(k))
    mintemp = min(temp)
    minidx = temp.index(mintemp)
    for idx, height in enumerate(ball_z[k:]):
        if height > mintemp:
            temp[minidx] = height
            indices[minidx] = idx
            mintemp = min(temp)
            minidx = temp.index(mintemp)

    k_highest_pts = indices
    possible_shot_indices = list()

    for candidate_index in k_highest_pts:
        idx = candidate_index
        while idx > 0:
            # Continue backtracking until we find a point below 10ft where
            # ball height reaches a local minimum (Noise near top of trajectory
            # can create fake local minima, so we ignore minima above z=10ft)
            if (ball_z[idx] - ball_z[idx-1] > 0) or (ball_z[idx] > 10):
                idx -= 1
            else:
                break

        # The shooter we know took the shot could only have done it if they had possession
        # at the local minimum
        if np.sqrt((shooter_x[idx] - ball_x[idx])**2 + (shooter_y[idx] - ball_y[idx])**2) < 3:
            possible_shot_indices.append(idx)

    # If shooter shot ball twice(+) in same event, we assume the last shot is the correct one
    # since it has the most pre-shot information available for us to analyze
    if len(possible_shot_indices) > 0:
        return max(possible_shot_indices)
    else:
        return None


def get_shot_index_old(movement):
    ball_z = movement[5]
    index_highest = ball_z.index(max(ball_z)) # Finds location of highest point of ball during event

    index_shot = index_highest
    while index_shot > 0:
        # Continue backtracking until we find a point below 10ft where ball height reaches a local minimum
        # (Noise near top of trajectory can create fake local minima)
        if (ball_z[index_shot] - ball_z[index_shot-1] > 0) or (ball_z[index_shot] > 10):
            index_shot -= 1
        else:
            break

    return index_shot


def get_catch_index(movement):
    shooter_x = movement[1]
    shooter_y = movement[2]
    ball_x = movement[3]
    ball_y = movement[4]
    shot_index = get_shot_index(movement)
    catch_index = shot_index
    while catch_index > 0:
        idx = catch_index - 1
        if np.sqrt((shooter_x[idx] - ball_x[idx])**2 + (shooter_y[idx] - ball_y[idx])**2) < 3:
            catch_index -= 1
        else:
            break

    return catch_index

def shooter_movement_between_frames(movement, shooterID, f1, f2):
    shooter_x = movement[1][f1:f2]
    shooter_y = movement[2][f1:f2]

    return list(zip(shooter_x, shooter_y))

def shooter_velocity_between_frames(movement, shooterID, f1, f2):
    shooter_movement = shooter_movement_between_frames(movement, shooterID, f1, f2)

    distance_sum = 0

    for frame in range(len(shooter_movement)-1):
        loc1 = shooter_movement[frame]
        loc2 = shooter_movement[frame+1]

        v = (loc2[0]-loc1[0], loc2[1]-loc1[1])
        distance_sum += np.sqrt(v[0]**2 + v[1]**2)

    avg_velocity = distance_sum / (0.04*(f2-f1))

    return avg_velocity


def is_catch_and_shoot(movement, shooterID):
    shot_index = get_shot_index(movement)
    catch_index = get_catch_index(movement)

    ballz = movement[5][catch_index:shot_index]
    return all(z > 1 for z in ballz)

def catchandshoot(Data, eventID, shooterID):
    secondscaught=frames_caught(Data, eventID, shooterID)
    movement=get_movements(Data, eventID, shooterID)
    miss_make=movement[0]
    index_highest=movement[5].index(max(movement[5]))
    index_shot=index_highest
    while True:
        if movement[5][index_shot]- movement[5][index_shot-1] > 0:
            index_shot=index_shot-1
        else:
            break
    ballz=movement[5][index_shot-secondscaught:index_shot]
    if (all(i > 1 for i in ballz)):
        return True
    else:
        return False
    
    
    
    
    
# begin some plotting catch and shoot data
Data = readJson('0021500440')
eventID = 32
shooterID = 200755
movement_data = get_movements(Data,eventID,shooterID)
shooter_x = movement_data[1]
shooter_y = movement_data[2]
ball_x = movement_data[3]
ball_y = movement_data[4]
ball_z = movement_data[5]
caught_index = get_catch_index(movement_data)
shot_index = get_shot_index(movement_data)
    
# plotting player x y and court
plt.figure()
court = plt.imread("fullcourt.png")
plt.figure(figsize=(15, 11.5))
plt.xlabel("x coordinate (ft)", fontsize=20)
plt.ylabel("y coordinate (ft)", fontsize=20)
plt.scatter(shooter_x[120:260],shooter_y[120:260], s=5)
plt.scatter(shooter_x[caught_index],shooter_y[caught_index], s=20,c = 'k')
plt.scatter(shooter_x[shot_index],shooter_y[shot_index], s=20, c = 'r')
plt.imshow(court, zorder=0, extent=[0,94,50,0])
plt.xlim(0,95)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# plotting ball x y and court 
plt.figure()
court = plt.imread("fullcourt.png")
plt.figure(figsize=(15, 11.5))
plt.xlabel("x coordinate (ft)", fontsize=20)
plt.ylabel("y coordinate (ft)", fontsize=20)
plt.scatter(ball_x[120:260],ball_y[120:260], s=5)
plt.scatter(ball_x[caught_index],ball_y[caught_index], s=20,c = 'k')
plt.scatter(ball_x[shot_index],ball_y[shot_index], s=20, c = 'r')
plt.imshow(court, zorder=0, extent=[0,94,50,0])
plt.xlim(0,95)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# plotting ball z
plt.figure()
plt.xlabel("time (frames from start of play)", fontsize=18)
plt.ylabel("z coordinate (ft)", fontsize=18)
plt.scatter(range(120,260), ball_z[120:260], s=5)
plt.scatter(caught_index, ball_z[caught_index], s=20,c = 'k')
plt.scatter(shot_index, ball_z[shot_index], s=20, c = 'r')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
