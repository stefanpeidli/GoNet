import sqlite3
import numpy as np
import io
import TrainingDataFromSgf as td

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

def upgradeTable(dbNameDist):
    con = sqlite3.connect(r"DB's/DistributionDB's/" + dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from test")
    data = cur.fetchone()
    db_size = data[0]
    for i in range(db_size):
        k = int(i)
        cur.execute("select * from test where id = ?", (k+1,))
        current = cur.fetchall()
        count = sum(current[0][2])
        for j in range(count):
            cur.execute("insert into test values (?, ?, ?)", (None, current[0][1], current[0][2]))
        con.commit()
    con.close()


def createDB():
    td.TrainingDataSgfPass(folder="dgs", id_list='dan_data_10', dbNameDist="dan_data_10")
    dbNameDist = 'dan_data_10'
    upgradeTable(dbNameDist)
    con= sqlite3.connect(r"DB's/DistributionDB's/" + dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from test where board = ?", (np.zeros(81, dtype=np.int32),))
    data = cur.fetchall()
    print(data)
    con.close()

createDB()

