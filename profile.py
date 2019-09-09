import pandas as pd
import pandas_profiling


xplane = pd.read_csv('datasets/xplane_8154.csv', delimiter=',')

print(xplane.describe())

profile_train = xplane.profile_report(title='XPLANE 8154 REPORT')

profile_train.to_file(output_file="xplane_8154.html")