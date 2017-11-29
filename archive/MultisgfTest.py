from TrainingDataFromSgf import *


dic = importTrainingData('Test_files')
count0 = 0
count3 = 3

for entry in dic:
    if np.linalg.norm(dic[entry]) == 0:
        count0 += 1
    elif np.linalg.norm(dic[entry]) == 3:
        count3 += 1
    else:
        print ('\n', entry, '\n', dic[entry], '\n')

print('0: ', count0, '\n', '3: ', count3)
