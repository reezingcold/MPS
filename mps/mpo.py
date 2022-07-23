'''
introducion: this package is created for the construction of MPO
ps: this package depends on numpy(https://numpy.org)
'''

class MPO(object):
    def __init__(self):
        self.mpo = None

    def get_data(self, site = "all"):
        return self.mpo if site == "all" else self.mpo[site]
    
    def get_dim(self, site):
        return self.mpo[site].shape
    
    @property
    def site_number(self):
        return len(self.mpo)