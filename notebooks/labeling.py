import numpy as np
import scipy.stats

class LabelingMechanism():
    def __init__(self,
                 propensity_attributes,
                 propensity_attributes_signs,
                 min_prob=0.2,
                 max_prob=0.8):
        assert len(propensity_attributes) == len(propensity_attributes_signs), "size of attributes and signs must be same"
        self.propensity_attributes = np.array(propensity_attributes)
        self.nb_propensity_attributes = len(propensity_attributes)
        self.propensity_attributes_signs = np.array(propensity_attributes_signs)
        self.min_prob = min_prob
        self.max_prob = max_prob
        
    def fit(self, x):
        x_e = x[:, self.propensity_attributes] * self.propensity_attributes_signs
        self.minx = x_e.min(0)
        self.maxx = x_e.max(0)
        
    def propensity_score(self, x):
        x_e = x[:, self.propensity_attributes] * self.propensity_attributes_signs
        scaled = self.min_prob + (x_e - self.minx) / (self.maxx - self.minx) * (self.max_prob - self.min_prob)
        return (scaled**(1 / self.nb_propensity_attributes)).prod(1)
    
def label_frequency(x, y, mean, cov, lm):
    score = lm.propensity_score(x[y==1])
    proba = scipy.stats.multivariate_normal(mean,cov).pdf(x[y==1])
    return score * proba / x[y==1].shape[0]
    