import numpy as np
from tqdm import tqdm
import csv

from db_connection import Neo4jConnection

class BaseImporter2:

    def __init__(self, argv):
        self.db = argv[3]
        self.connection = Neo4jConnection(uri=argv[0], user=argv[1], pwd=argv[2])
class BaseImporter:

    def __init__(self, file, argv):
        self.db = argv[3]
        self.connection = Neo4jConnection(uri=argv[0], user=argv[1], pwd=argv[2])
        self.file = file
    
    def get_rows(self, delimiter='\t'):
        print('Reading rows...')
        file_data = open(self.file, 'r')
        reader = csv.DictReader(file_data, delimiter=delimiter)
        list_data = []
        for row in reader:
            list_data.append(row)
        
        return list_data
    
    def get_file_size(self, delimiter='\t'):
        print('Getting file size...')
        file_data = open(self.file, 'r')
        reader = csv.DictReader(file_data, delimiter=delimiter)
        list_data = []
        for row in reader:
            list_data.append(row)
        
        return len(list_data)

    def load_in_batch(self, query, data, size, chunk_size=100):
        print('Start loading data...')
        chunks_num = size / chunk_size
        if chunks_num < 1:
            self.connection.query(query, parameters={'rows': data}, db=self.db)
        else:
            batches = [x.tolist() for x in np.array_split(data, chunks_num)]
            for b in tqdm(batches):
                self.connection.query(query, parameters={'rows': b}, db=self.db)
