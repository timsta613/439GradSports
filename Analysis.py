import kmedoids
import numpy as np
import pandas as pd

from itertools import combinations

from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn import cluster
from sklearn.metrics import pairwise_distances


def cumfreq(a, numbins=10, defaultreallimits=None):
	h,b,_ = plt.hist(a, numbins, defaultreallimits)
	cumhist = np.cumsum(h*1, axis=0)
	cumhist = [x / max(cumhist) for x in cumhist]
	return cumhist, b[:-1]


def twoSampleKS(sample1, sample2, sigThresh=0.05):
	n1 = len(sample1)
	n2 = len(sample2)

	xmax = max(np.nanmax(sample1), np.nanmax(sample2))
	xmin = min(np.nanmin(sample1), np.nanmin(sample2))

	bins = np.linspace(xmin, xmax, 200)
	hist1,_,_ = plt.hist(sample1, bins)
	hist2,_,_ = plt.hist(sample2, bins)
	cumhist1 = np.cumsum(hist1*1, axis=0)
	cumhist2 = np.cumsum(hist2*1, axis=0)
	cdf1 = [x / max(cumhist1) for x in cumhist1]
	cdf2 = [x / max(cumhist2) for x in cumhist2]

	maxdiff = max(np.abs(cdf1[i] - cdf2[i]) for i in range(len(bins)-1))
	threshold = np.sqrt(-0.5*np.log(sigThresh))*np.sqrt((n1+n2)/(n1*n2))
	print('N1={}, N2={}, D={}'.format(n1, n2, maxdiff))
	p = np.exp(-2*(maxdiff**2)*n1*n2/(n1+n2))

	return maxdiff > threshold, p, maxdiff

def speedAnalysis():
	sbc_headers = {x: 'SBC{:.1f}s'.format(x) for x in [y/2.0 for y in range(1,11)]}
	sbs_headers = {x: 'SBS{:.1f}s'.format(x) for x in [y/2.0 for y in range(1,11)]}

	X = pd.read_csv('output/3pt_speeds.csv', header=0)
	X['decel'] = X['speedFromCatchToShoot'] - X['SBC0.5s']
	CS = X.loc[X['isCatchAndShoot']==True]
	NonCS = X.loc[X['isCatchAndShoot']==False]

	time_intervals = sbc_headers.keys()

	# Find speeds associated with makes/misses on catch-and-shoots (C&S) based on pre-catch speeds
	for t in time_intervals:
		header = sbc_headers[t]

		makes = CS.loc[CS['isMake']==True][header]
		makes = makes[~np.isnan(makes)]

		misses = CS.loc[CS['isMake']==False][header]
		misses = misses[~np.isnan(misses)]

		bins = np.linspace(0, 20, 80)

		plt.clf()
		makehist = plt.hist(makes, bins, alpha=0.5, label='Makes')
		misshist = plt.hist(misses, bins, alpha=0.5, label='Misses')
		plt.legend(loc='upper right')
		plt.title('Speed from {} seconds before catch to catch'.format(t))
		plt.xlabel('Shooter average speed (ft/s)')
		plt.ylabel('Frequency')
		plt.savefig('output/plots/speedBeforeCatch{}s.png'.format(t))
		plt.savefig('output/plots/speedBeforeCatch{}s.eps'.format(t))

		confidence = 5
		s, p, D = twoSampleKS(makes, misses, 0.01*confidence)
		print(p)
		print(D)
		print()

		if s: print('Difference in pre-catch speed ({}s before catch) on C&S makes vs misses is significant @ {}% confidence level!'.format(
			t, confidence))

	print('\nNow speed before shooting...\n')

	# Find speeds associated with makes/misses on non-C&S based on pre-shot speeds

	for t in time_intervals:
		header = sbs_headers[t]

		makes = NonCS.loc[NonCS['isMake']==True][header]
		makes = makes[~np.isnan(makes)]

		misses = NonCS.loc[NonCS['isMake']==False][header]
		misses = misses[~np.isnan(misses)]

		bins = np.linspace(0, 20, 80)

		plt.clf()
		makehist = plt.hist(makes, bins, alpha=0.5, label='Makes')
		misshist = plt.hist(misses, bins, alpha=0.5, label='Misses')
		plt.legend(loc='upper right')
		plt.title('Speed from {} seconds before shot to shot'.format(t))
		plt.xlabel('Shooter average speed (ft/s)')
		plt.ylabel('Frequency')
		plt.savefig('output/plots/speedBeforeShot{}s.png'.format(t))
		plt.savefig('output/plots/speedBeforeShot{}s.eps'.format(t))

		confidence = 10
		s, p, D = twoSampleKS(makes, misses, 0.01*confidence)

		print(p)
		print(D)
		print()

		if s: print('Difference in pre-shot speed ({} s before shot) on non-C&S makes vs misses is significant @ {}% confidence level'.format(
			t, confidence))

	print('\nNow speed between catching and shooting...\n')

	# Find speeds associated with makes/misses on all shots based on speed between making catch and taking shot

	makes = X.loc[X['isMake']==True]['speedFromCatchToShoot']
	makes = makes[~np.isnan(makes)]

	misses = X.loc[X['isMake']==False]['speedFromCatchToShoot']
	misses = misses[~np.isnan(misses)]

	bins = np.linspace(0, 20, 80)

	plt.clf()
	makehist = plt.hist(makes, bins, alpha=0.5, label='Makes')
	misshist = plt.hist(misses, bins, alpha=0.5, label='Misses')
	plt.legend(loc='upper right')
	plt.title('Speed between catch and shot')
	plt.xlabel('Shooter average speed (ft/s)')
	plt.ylabel('Frequency')
	plt.savefig('output/plots/speedCatchToShoot.png')
	plt.savefig('output/plots/speedCatchToShoot.eps')

	confidence = 10
	s, p, D = twoSampleKS(makes, misses, 0.01*confidence)

	print(p)
	print(D)
	print()

	if s: print('Difference in speed between catch and shot on all makes vs all misses is significant @ {}% confidence level'.format(confidence))

	# Find speeds associated with makes/misses on all 

	makes = CS.loc[CS['isMake']==True]['speedFromCatchToShoot']
	makes = makes[~np.isnan(makes)]

	misses = CS.loc[CS['isMake']==False]['speedFromCatchToShoot']
	misses = misses[~np.isnan(misses)]

	bins = np.linspace(0, 20, 80)

	plt.clf()
	makehist = plt.hist(makes, bins, alpha=0.5, label='Makes')
	misshist = plt.hist(misses, bins, alpha=0.5, label='Misses')
	plt.legend(loc='upper right')

	plt.title('Speed between catch and shot for Catch & Shoot 3s')
	plt.xlabel('Shooter average speed (ft/s)')
	plt.ylabel('Frequency')
	plt.savefig('output/plots/speedCatchToShootCNS.png')
	plt.savefig('output/plots/speedCatchToShootCNS.eps')

	confidence = 10
	s, p, D = twoSampleKS(makes, misses, 0.01*confidence)

	print(p)
	print(D)
	print()

	if s: print('Difference in speed between catch and shot on C&S makes vs misses is significant @ {}% confidence level\n'.format(confidence))


	makes = NonCS.loc[NonCS['isMake']==True]['speedFromCatchToShoot']
	makes = makes[~np.isnan(makes)]

	misses = NonCS.loc[NonCS['isMake']==False]['speedFromCatchToShoot']
	misses = misses[~np.isnan(misses)]

	bins = np.linspace(0, 20, 80)

	plt.clf()
	makehist = plt.hist(makes, bins, alpha=0.5, label='Makes')
	misshist = plt.hist(misses, bins, alpha=0.5, label='Misses')
	plt.legend(loc='upper right')

	plt.title('Speed between catch and shot for Catch & Shoot 3s')
	plt.xlabel('Shooter average speed (ft/s)')
	plt.ylabel('Frequency')
	plt.savefig('output/plots/speedCatchToShootNonCNS.png')
	plt.savefig('output/plots/speedCatchToShootNonCNS.eps')

	confidence = 10
	s, p, D = twoSampleKS(makes, misses, 0.01*confidence)

	print(p)
	print(D)
	print()

	if s: print('Difference in speed between catch and shot on non-C&S makes vs misses is significant @ {}% confidence level'.format(confidence))

	'''

	makes = CS.loc[CS['isMake']==True]['decel']
	makes = makes[~np.isnan(makes)]

	misses = CS.loc[CS['isMake']==False]['decel']
	misses = misses[~np.isnan(misses)]

	bins = np.linspace(-10, 10, 80)

	plt.clf()
	makehist = plt.hist(makes, bins, alpha=0.5, label='Makes')
	misshist = plt.hist(misses, bins, alpha=0.5, label='Misses')
	plt.legend(loc='upper right')

	plt.title('C&S speed reduction after catch')
	plt.xlabel('Shooter change in average speed (ft/s)')
	plt.ylabel('Frequency')
	plt.savefig('output/plots/deceleration.png')
	plt.savefig('output/plots/deceleration.eps')

	confidence = 10
	s, p, D = twoSampleKS(makes, misses, 0.01*confidence)

	print(p)
	print(D)
	print()

	if s: print('Difference in speed reduction after catch for C&S makes vs misses is significant @ {}% confidence level'.format(confidence))

	'''

FEATURES_OF_INTEREST = [
	'ShotClock', 'CnS', 'Points', 'TimeCns', 'v_CnS', 'dist_at_shot', 'shot_angle', 'shooter_move_angle',
	'shooter_travel', 'def_dist_at_shot', 'def_avg_vel', 'SpeedBeforeCatch0.5s']

REDUCED_FEATURES = [
	'def_dist_at_shot', 'shooter_travel', 'speedBeforeCatch0.5s', 'dist_at_shot', 'TimeCns']

CNS_FEATURES = [
	'ShotClock', 'Points', 'TimeCns', 'v_CnS', 'dist_at_shot', 'shot_angle', 'shooter_move_angle',
	'shooter_travel', 'def_dist_at_shot', 'def_avg_vel', 'SpeedBeforeCatch0.5s']

OTHER_FEATURES = [
	'ShotClock', 'Points', 'TimeCns', 'v_CnS', 'dist_at_shot', 'shot_angle', 'shooter_move_angle',
	'shooter_travel', 'def_dist_at_shot', 'def_avg_vel']

def generatePlayerMeans(threshold):
	X = pd.read_csv('threes.csv', header=0)
	shooters = set(X['ShooterID'])

	cnsCols = ['CnS_{}'.format(name) for name in CNS_FEATURES]
	otherCols = ['nonCnS_{}'.format(name) for name in OTHER_FEATURES]
	featureCols = ['shooterID', 'pctCnS']
	allCols = cnsCols + otherCols + featureCols

	shooterMeans = pd.DataFrame(columns=allCols)

	for shooterID in shooters:
		shooterX = X.loc[X['ShooterID']==shooterID]
		if len(shooterX) >= threshold:

			shooterCnS = shooterX.loc[shooterX['CnS']==True]
			shooterCnS = shooterCnS[CNS_FEATURES]
			shooterCnS.columns = cnsCols

			shooterOther = shooterX.loc[shooterX['CnS']==False]
			shooterOther = shooterOther[OTHER_FEATURES]
			shooterOther.columns = otherCols

			meanCnS = np.mean(shooterCnS)
			meanOther = np.mean(shooterOther)
			moreFeatures = pd.Series({
				'shooterID': shooterID,
				'pctCnS': len(shooterCnS) / (len(shooterCnS) + len(shooterOther))
				})
			shooterMean = pd.concat([moreFeatures, meanCnS, meanOther], axis=0)
			# print(shooterMean)
			if not shooterMean.hasnans:
				shooterMeans = shooterMeans.append(shooterMean, ignore_index=True)

	return shooterMeans

def clusterPlayerMeans(shooterMeans, max_clusters=10):
	clusterings = list()
	performances = list()

	for k in range(1, max_clusters+1):
		print(k)
		kmeans = cluster.KMeans(n_clusters=k)
		labels = kmeans.fit_predict(shooterMeans)
		wgss = 0
		for label in range(k):
			indices = np.nonzero(labels==label)[0]
			for combo in combinations(indices, 2):
				v1 = shooterMeans.loc[combo[0]]
				v2 = shooterMeans.loc[combo[1]]
				d = [v1[name] - v2[name] for name in shooterMeans.columns]
				dist = np.linalg.norm(d)**2
				wgss += dist

		clusterings.append(labels)
		performances.append(wgss)

	return clusterings, performances

def clusterKMedoids(shooterMeans, max_clusters=10):
	medoids = list()
	clusterings = list()
	performances = list()
	n, _ = shooterMeans.shape
	D = pairwise_distances(shooterMeans)

	for k in range(1,max_clusters+1):
		print(k)
		m, c = kmedoids.kMedoids(D, k, 10000)

		medoids.append(m)

		# Cluster output of kMedoids is dictionary {cid: [x where c(x)==cid]}
		# Transform to list of cluster id for similarity with kMeans
		labels = [-1] * n
		for label in c:
			for idx in c[label]:
				labels[idx] = label

		labels = np.array(labels) # Transform so that np.nonzero works

		wgss = 0
		for label in range(k):
			indices = np.nonzero(labels==label)[0]
			for combo in combinations(indices, 2):
				v1 = shooterMeans.loc[combo[0]]
				v2 = shooterMeans.loc[combo[1]]
				d = [v1[name] - v2[name] for name in shooterMeans.columns]
				dist = np.linalg.norm(d)**2
				wgss += dist

		clusterings.append(labels)
		performances.append(wgss)

	return clusterings, performances, medoids


def plotClusters(shooterMeans, clusters=None, method='MDS', labels=None):
	transformer = None
	if method == 'MDS':
		transformer = manifold.MDS()
	elif method == 'TSNE':
		transformer = manifold.TSNE()
	else:
		raise ValueError('Method can be either \'MDS\' or \'TSNE\'')

	features = list(shooterMeans.columns)
	features.remove('shooterID')

	X = transformer.fit_transform(shooterMeans[features])

	x = [row[0] for row in X]
	y = [row[1] for row in X]
	fig, ax = plt.subplots()
	if clusters is not None:
		ax.scatter(x, y, c=clusters)
	else:
		ax.scatter(x, y)

	# Labels specify which players to identify with their shooter ID in the plot
	# Will replace with name in the future
	for idx in labels:
		row = shooterMeans.loc[idx]
		ax.annotate(int(row['shooterID']), (x[idx], y[idx]))

	plt.show()

def main():
	D = generatePlayerMeans(60)

	features = list(D.columns)
	features.remove('shooterID') # DON'T RUN CLUSTERING ON PLAYER ID!

	max_clusters = 10

	c, p = clusterPlayerMeans(D[features], max_clusters=max_clusters)
	cm, pm, m = clusterKMedoids(D[features], max_clusters=max_clusters)

	# plt.plot(list(range(1, max_clusters+1)), pm)
	# plt.show()

	# Seems that 3 clusters is the best (use k-medoids)
	plotClusters(D, clusters=cm[2], labels=m[2])

	return D, c, p, cm, pm, m

if __name__ == '__main__':
	main()