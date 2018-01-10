import sqlite3
import numpy as np
import io

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

def createTable(dbNameDist):
    con = sqlite3.connect(r"DB's/DistributionDB's/" + dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from test")
    data = cur.fetchall()
    db_size = data[0][0]
    cur.execute("create table dist as select * from test")
    cur.execute("select * from dist where id = 250")
    data = cur.fetchall()
    print(data)
    for i in range(db_size):
        cur.execute("select * from test where id = ?", (i+1,))
        current = cur.fetchall()
        count = sum(current[0][2])
        for j in range(count+1):
            cur.execute("insert into dist values (?, ?, ?)", (None, current[0][1], current[0][2]))
    con.close()


def test():
    dbNameDist = 'dan_data_10_test'
    createTable(dbNameDist)
    con= sqlite3.connect(r"DB's/DistributionDB's/" + dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select * from dist where id > 5700")

#test()
