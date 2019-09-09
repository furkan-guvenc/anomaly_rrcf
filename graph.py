import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/xplane_8154.csv', delimiter='|')
print(df.describe())
"""
df = df.drop(
    columns=['ACID', 'APUF', 'BPGR_1', 'BPYR_2', 'CALT', 'DVER_1', 'DVER_2', 'ESN_1', 'ESN_2', 'ESN_3', 'ESN_4', 'EVNT',
             'FADF', 'FADS',
             'FIRE_1', 'FIRE_2', 'FIRE_3', 'FIRE_4', 'FQTY_1', 'FQTY_2', 'FQTY_3', 'FQTY_4', 'GPWS', 'HF1', 'HF2',
             'POVT', 'PUSH', 'SMKB', 'SMOK',
             'SNAP', 'TMAG', 'VAR_1107', 'VAR_2670', 'VAR_5107', 'VAR_6670', 'WAI_1', 'WAI_2', 'WSHR', 'DATE_DAY',
             'DATE_MONTH', 'DATE_YEAR', 'MW',
             'NSQT', 'APFD', 'MRK', 'VHF1', 'VHF2', 'VHF3', 'LGDN', 'LGUP', 'SHKR', 'MSQT_1', 'MSQT_2', 'GMT_HOUR',
             'GMT_MINUTE', 'GMT_SEC',
             'ACMT', 'TOCW', 'WOW', 'N1T', 'N1C', 'A_T', 'TCAS', 'TAI', 'ATEN'])"""
print("dataset finished")
for i, col in enumerate(df.columns):
    plt.title(col)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.plot(df[col])
    plt.savefig('outputs/{}.png'.format(col))
    plt.clf()
    print("{}:{} plotted".format(i, col))

exit()