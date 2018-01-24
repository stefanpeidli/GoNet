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

def expand_db(dbNameDist):
    con = sqlite3.connect(r"DB/Dist/" + dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from movedata")
    data = cur.fetchone()
    db_size = data[0]
    for i in range(db_size):
        k = int(i)
        cur.execute("select * from movedata where id = ?", (k+1,))
        current = cur.fetchone()
        count = sum(current[2])
        for j in range(count-1):
            cur.execute("insert into movedata values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (None, current[1], current[2], current[3], current[4], current[5], current[6], current[7],
                         current[8],current[9], current[10]))
        con.commit()
    con.close()
# TODO: debugging: filtered should be array of ints not doubles


def createDB(id_list, db_name, mode='dist', expandDB=False):
    if mode == 'dist':
        td.TrainingDataSgfPass(folder="dgs", id_list=id_list, dbNameDist=db_name)
        if expandDB:
            expand_db(db_name)
    elif mode == 'move':
        td.TrainingDataSgfPass(folder="dgs", id_list=id_list, dbNameMoves=db_name)
    else:
        print('please select a valid mode: \"dist\" or \"move\"')


'''
readme:
for creating databases there are several options which need to be configured before calling createDB:
1. choose an id_list (list of id's which will be extracted from the dgs folder
2. choose a name for your new db
3. aditional argument: choose mode = 'move' if you want to create a db with (board,dirac-distributions) as fundamental
data structure
4. additional argument: choose expandDB = True if you want to additionally enrich the database by copying the
distributions as often as the board was played
'''
#createDB('dan_data_1', 'dan_data_1_expanded', expandDB=True)

def test():
    db_name = "dan_data_10"
    con = sqlite3.connect(r"DB/Dist/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from movedata")
    cur.fetchall()
    data = con.close()
    print(data[0])

test()