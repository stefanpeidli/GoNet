import sqlite3

con = sqlite3.connect(r"DB's/DistributionDB's/" + 'dan_data_10', detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("select count(*) from test")
data = cur.fetchall()
datasize = data[0][0]
con.close()

print(datasize)

with open("guru.txt","w+") as f:
    f.write(str(datasize))
    f.close()

