import numpy as np 
from pymoo.indicators.distance_indicator import DistanceIndicator, euclidean_distance

class S(DistanceIndicator):
	def __init__(self, pf, **kwargs):
		self.pf = pf  # set of points whose Spread metric is to be computed
		#super().__init__(pf, euclidean_distance, 1, **kwargs)
	def do(self): 	
		def determineDistanceToNearestNeighbor(f):
			i = f[0]  # index of required point in original set
			f = f[1:]  # extracts the required  point 		 
			dist =  euclidean_distance(f, self.pf, norm = 1)   # computes euclidean distance
			dist[i.astype(int)] = np.inf # sets distance of a point to itself to an infeasible value
			dist = np.min(dist) 	# determines distance to the nearest neighbor  
			return dist  # returns the distance of f to the nearest point/neigbor in the set self.pf
		arr =  np.array(np.arange(self.pf.shape[0])) 
		f= np.column_stack( (arr, self.pf)) 	 
		dst = np.apply_along_axis(determineDistanceToNearestNeighbor, 1 , f) # array of distances to the nearest neigbors
		return np.mean(dst)	 