import csv
import os
import random
import shutil
import subprocess

from Data import movementDataSample
from Utils import *


# Change these paths to your local instances before running
dropboxDir = '/Users/tzhong/Dropbox (MIT)/6_439 Group Project'
codeDir = '/Users/tzhong/Code/Classes/2018/Fall/ids-131/439GradSports'


def main():
	extractSpecificMovementData(movementDataSample)
	writeSpeedCSV()


def extractSpecificMovementData(fnames):
	zipDir = os.path.join(dropboxDir, 'Data/nba-movement-data-master/data/')
	csvDir = os.path.join(dropboxDir, 'Data/events/')

	movementDataDir = os.path.join(codeDir, 'data/movement/')
	eventDataDir = os.path.join(codeDir, 'data/events/')

	os.chdir(zipDir)
	targetFiles = os.listdir(movementDataDir)
	print(targetFiles)

	for file in fnames:
		print('\n{}'.format(file))

		# Output of '7z l {filename}' in command line (only way to see filenames inside .7z without extracting)
		file_contents = subprocess.check_output(['7za','l',file]).decode('utf-8')
		gameID = [x[:x.find('.json')] for x in file_contents.split(' ') if (x.find('.json') > 0)][0]

		print(gameID)
		if '{}.json'.format(gameID) not in targetFiles:
			os.system('7za x {} -o{}'.format(
				file, movementDataDir))
			print('{} not found!'.format(os.path.join(movementDataDir, '{}.json'.format(gameID))))
		else:
			print('File {} (Game ID {}) is already in the data folder'.format(file, gameID))

	gameIDs = [x.split('.')[0] for x in os.listdir(movementDataDir)]

	print(csvDir)
	for gameID in gameIDs:
		csvfname = '{}.csv'.format(gameID)
		csvpath = os.path.join(csvDir, csvfname)
		try:
			shutil.copy(csvpath, eventDataDir)
		except FileNotFoundError:
			print('File not found: {}'.format(csvpath))

	return gameIDs


def extractRandomMovementData(numFiles):
	zipDir = os.path.join(baseDir, 'Data/nba-movement-data-master/data/')
	csvDir = os.path.join(baseDir, '/Data/events/')

	movementDataDir = os.path.join(codeDir, 'data/movement/')
	eventDataDir = os.path.join(codeDir, '/data/events/')

	os.chdir(zipDir)
	fileList = os.listdir()

	fileSubset = random.sample(fileList, numFiles)

	for file in fileSubset:
		os.system('7za x {} -o{}'.format(
			file, movementDataDir))

	gameIDs = [x.split('.')[0] for x in os.listdir(movementDataDir)]

	for gameID in gameIDs:
		csvfname = '{}.csv'.format(gameID)
		try:
			shutil.copy(os.path.join(csvDir, csvfname), eventDataDir)
		except FileNotFoundError:
			continue

	return fileSubset, gameIDs


def writeSpeedCSV():
	os.chdir(codeDir)
	gameIDs = [x.split('.')[0] for x in os.listdir('./data/movement/')] # Combs through every movement data file in data directory
	outfile = 'test/3pt_speeds.csv'

	for gameID in gameIDs:
		print(gameID)
		Data = readJson(gameID)

		all3pts = get_all_3pt(Data)

		for shot in all3pts:
			shooterID = shot['shooterID']
			eventID = shot['eventID']
			movement = get_movements(Data, eventID, shooterID)
			if movement is None:
				continue

			t_shot = get_shot_index(movement) # Time in frames from beginning of the event
			if t_shot is None:
				continue

			t_catch = get_catch_index(movement) # Same

			t_min_before_highest = get_shot_index_old(movement)
			if t_shot == t_min_before_highest:
				continue
			else:
				print('Event {} - Previously thought Shooter {} caught/shot @ {} but now caught @ {} / shot @ {}'.format(
					eventID, shooterID, t_min_before_highest, t_catch, t_shot))

			if t_catch == t_shot:
				print('Event {} - Shooter {} caught ball at {} and shot at {}'.format(
					eventID, shooterID, t_catch, t_shot))
				continue

			is_cns = is_catch_and_shoot(movement, shooterID)

			t_with_ball = (t_shot - t_catch) / 25

			v_with_ball = shooter_velocity_between_frames(movement, shooterID, t_catch, t_shot)

			v_before_shot = dict()
			# Find average velocity over n secs before shot for n = 0.5, 1, 1.5, ... 5]
			for n in [x / 2.0 for x in range(1, 11)]:
				v_before_shot[n] = None
				if n < t_with_ball:
					v_before_shot[n] = shooter_velocity_between_frames(
						movement, shooterID, int(t_shot - 25*n), t_shot)
				# print('{} - {}'.format(n, v_before_shot[n]))


			v_before_catch = dict()
			for n in [x / 2.0 for x in range(1, 11)]:
				v_before_catch[n] = None
				start_frame = int(t_catch - 25*n)
				if start_frame > 0:
					v_before_catch[n] = shooter_velocity_between_frames(
						movement, shooterID, start_frame, t_catch)

			row = list()
			row.append(gameID)
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
			row.append(closest_defender_dist(movement, t_shot, Data)[0])
			row.append(closest_defender_dist(movement, t_shot, Data)[1])
			row.append(closest_defender_velocity(movement,t_catch,t_shot, Data))
			[row.append(v_before_shot[x]) for x in v_before_shot]
			[row.append(v_before_catch[x]) for x in v_before_catch]

			with open(outfile, 'a+', newline='') as outf:
				writer = csv.writer(outf)
				writer.writerow(row)


if __name__ == '__main__':
	main()
