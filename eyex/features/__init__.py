import numpy as np
from scipy.spatial import ConvexHull

class Standard():
    
    def __init__(self, data, fixations_idx_list, saccades_idx_list):
        self.data = data
        self.fixations_idx_list = fixations_idx_list
        self.saccades_idx_list = saccades_idx_list
        self.saccades_idx_list_filtered = [item for item in saccades_idx_list if len(item)>1]
        self.saccades_idx_list_filtered2 = [item for item in saccades_idx_list if len(item)>2]
        
    def fixation_lengths(self):
        lengths = []
        for elem in self.fixations_idx_list:
            lengths.append(len(elem))
        lengths = np.array(lengths)
        return lengths
    
    def fixation_number(self):
        return len(self.fixations_idx_list)
    
    def fixation_centers(self):
        centers = np.zeros((len(self.fixations_idx_list), 2))
        for i, elem in enumerate(self.fixations_idx_list):
            centers[i] = np.mean(self.data[elem], axis = 0)
        return centers 
    
    def fixation_elementary_amplitudes(self):
        elementary_amplitudes = []
        for elem in self.fixations_idx_list:
            fixation = self.data[elem]
            n = len(fixation)
            elementary_amplitudes.append(np.linalg.norm(fixation[:n-1] - fixation[1:], axis = 1))
        elementary_amplitudes = np.array(elementary_amplitudes)
        return elementary_amplitudes
    
    def fixation_integral_amplitudes(self):
        fixation_integral_amplitudes = []
        for elem in self.fixations_idx_list:
            fixation = self.data[elem]
            fixation_integral_amplitudes.append(np.sum(np.linalg.norm(fixation[1:] - fixation[:-1], axis=1)))
        fixation_integral_amplitudes = np.array(fixation_integral_amplitudes)
        return fixation_integral_amplitudes
    
    def saccade_amplitides(self):
        amplitudes = []
        for elem in self.saccades_idx_list_filtered:
            amplitude = self.data[elem]
            amplitudes.append(np.linalg.norm(amplitude[0] - amplitude[-1]))
        amplitudes = np.array(amplitudes)
        return amplitudes
    
    def saccade_peak_amplitude(self):
        peak_amplitudes = []
        for elem in self.saccades_idx_list_filtered:
            saccade = self.data[elem]
            peak_amplitudes.append(np.max(np.linalg.norm(saccade[1:] - saccade[:-1], axis=1)))
        peak_amplitudes = np.array(peak_amplitudes)
        return peak_amplitudes
    
    def saccade_number(self):
         return len(self.saccades_idx_list)
        
    def saccade_integral_amplitudes(self):
        saccade_integral_amplitudes = []
        for elem in self.saccades_idx_list_filtered:
            saccade = self.data[elem]
            saccade_integral_amplitudes.append(np.sum(np.linalg.norm(saccade[1:] - saccade[:-1], axis=1)))
        saccade_integral_amplitudes = np.array(saccade_integral_amplitudes)
        return saccade_integral_amplitudes
    
    def saccades_centers(self):
        saccades_centers = np.zeros((len(self.saccades_idx_list), 2))
        for i, elem in enumerate(self.saccades_idx_list):
            saccades_centers[i] = np.mean(self.data[elem], axis = 0)
        return saccades_centers 
    
    def track_length(self):
        track_length = np.sum(np.linalg.norm(self.data[1:] - self.data[:-1], axis=1))
        return track_length
    
    def track_hull(self):
        return ConvexHull(self.data).volume
    
    def fixations_hull_vol(self):
        fixations_hull_vol = []
        for elem in self.fixations_idx_list:
            fixation = self.data[elem]
            hull = ConvexHull(fixation)
            fixations_hull_vol.append(hull.volume)
        fixations_hull_vol = np.array(fixations_hull_vol)
        return fixations_hull_vol
    
    def saccades_hull_vol(self):
        saccades_hull_vol = []
        for elem in self.saccades_idx_list_filtered2:
            saccade = self.data[elem]
            hull = ConvexHull(saccade)
            saccades_hull_vol.append(hull.volume)
        saccades_hull_vol = np.array(saccades_hull_vol)
        return saccades_hull_vol
        
class TDA():

    def __init__(self, data):
        self.data = data