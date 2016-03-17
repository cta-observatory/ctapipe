class CameraData():
    """
    A class containing CTA telescope data. The underlying datastructure is an astropy.table
    This table holds a chunk of events from ONE telescope. The exact contents of the
    events depends on the data model.

    """
    def __init__(self, table):
        self.table = table

    #overwrite the attribute access and just return the attribute of the underlying table
    #this way we can add more columns to the astropy.table without changing stuff in here
    def __getattr__(self, key):
        return self.table[key]

    def __len__(self):
        '''
        returns the number of events contained in this chunk.
        '''
        return len(self.table)

    @property
    def telescope_id(self):
        '''

        '''
        return self.table.meta['telescope_id']

    @property
    def telescope_name(self):
        return self.table.meta['telescope_name']

    @property
    def telescope_geometry(self):
        return self.table.meta['geometry']


    def __iter__(self):
        for row in self.table:
            yield row
