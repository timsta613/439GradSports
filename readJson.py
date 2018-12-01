import csv
import json


def main():
	gameID = '0021500440'
	readJson(gameID)

def readJson(gameID):
	movementfile = 'data/movement/{}.json'.format(gameID)
	with open(movementfile) as f:
		movement = json.load(f)

	movementsByEventID = {int(event['eventId']): event for event in movement['events']}

	movementByEvents = dict()

	eventsfile = 'data/events/{}.csv'.format(gameID)
	with open(eventsfile) as f:
		reader = csv.reader(f)
		for row in reader:
			try:
				eventID = int(row[1])
			except:
				print('Error with event: {}'.format(row))
				continue
			if eventID in movementsByEventID.keys() and len(movementsByEventID[eventID]['moments']) > 0:
				movementByEvents[eventID] = dict()
				movementByEvents[eventID]['eventData'] = row
				movementByEvents[eventID]['movementData'] = movementsByEventID[eventID]['moments']

	return movementByEvents

if __name__ == '__main__':
	readJson()