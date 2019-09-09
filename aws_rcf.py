import sagemaker
import urllib.request
import pandas as pd

f = 'nyc_taxi.csv'

taxi_data = pd.read_csv(f, delimiter=',')

print(taxi_data.head())

execution_role = sagemaker.get_execution_role()
rcf = sagemaker.RandomCutForest(role=execution_role, num_trees=100, num_samples_per_tree=256)
