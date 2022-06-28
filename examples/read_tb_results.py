# minimal example that shows how to read tensorboard results into one big pandas data frames

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

#----------------------------------------------------------------------------------------------
event_file = sorted(list(Path('runs').glob('exp-*/events.out.tfevents.*')))[-1]

ea = event_accumulator.EventAccumulator(str(event_file))
ea.Reload()

df = pd.DataFrame()

for i, scalar_tag in enumerate(ea.Tags()['scalars']):
    if i == 0:
        df = pd.DataFrame(ea.Scalars(scalar_tag)).rename({'value':scalar_tag}, axis = 1)
    else:
        # we have to drop the wall_time since it is slightly different for every scalar tag
        tmp_df = pd.DataFrame(ea.Scalars(scalar_tag)).rename({'value':scalar_tag}, axis = 1).drop(columns = 'wall_time')
        df = pd.merge(df, tmp_df, on = 'step')

print(df.columns)
print(df)

# to plot a metric / statistic vs step use:
# df.plot(x = 'step', y = 'Global_MEAN')
