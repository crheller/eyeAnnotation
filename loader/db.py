import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
from settings import MONGO_URL
import pymongo as pym
import pandas as pd
# db query.
def dbquery(fields, filt):
    """
    Return dataframe with fields for each dataset matching filter.
    filter is dictionary of query filters
    fields is list of strings
    """
    client = pym.MongoClient(MONGO_URL)
    db = client.rolidb
    collection = db.data
    documents = collection.find(filt)
    df = []
    for doc in documents:
        data = []
        for k in fields:
            data.append(doc[k])
        df.append(data)
    df = pd.DataFrame(columns=fields, data=np.array(df))
    return df