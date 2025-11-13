import pymongo

class MongoDB:
    my_client = pymongo.MongoClient('localhost', 27005)
    dbname = my_client['Drive-Monitoring']

    def getDailyZenith(self, T_min, T_max):
        result = list(self.dbname['Position'].find({'T': {'$gte': T_min, '$lt': T_max}}, {'_id': 0, 'Az': 0}).sort('T', 1))
        return result
	
    def getDamageValues(self, T_min, T_max):
        result = list(self.dbname['Damage'].find({'T': {'$gte': T_min, '$lt': T_max}}).sort('T', 1))
        return result