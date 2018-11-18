import csv
import json


def main():
	gameID = '0021500440'

	movementfile = 'data/movement/{}.json'.format(gameID)
	with open(movementfile) as f:
		movement = json.load(f)

	movementByEvents = dict()

	eventsfile = 'data/events/{}.csv'.format(gameID)
	with open(eventsfile) as f:
		reader = csv.reader(f)
		for eventID, row in enumerate(reader):
			movementByEvents[eventID] = dict()
			movementByEvents[eventID]['eventData'] = row
			movementByEvents[eventID]['movementData'] = movement['events'][eventID]

if __name__ == '__main__':
	main()