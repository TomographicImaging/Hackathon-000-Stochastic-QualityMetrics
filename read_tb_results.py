# minimal example that shows how to read tensorboard results into pandas data frames
#
# structure of the tensorboard experiment result folder created by the ImageQualityCallback 
# containing the local and global image metrics and statistics should look like below
#
# the first event file contains the cost. 
# all other contain the global/local image metrics and statistics
#
#
# |-- events.out.tfevents.1649262033.amun
# |-- global_metrics
# |   |-- MAE
# |   |   `-- events.out.tfevents.1649262033.amun
# |   |-- MSE
# |   |   `-- events.out.tfevents.1649262033.amun
# |   `-- PSNR
# |       `-- events.out.tfevents.1649262033.amun
# |-- global_statistics
# |   |-- MEAN
# |   |   `-- events.out.tfevents.1649262033.amun
# |   `-- STDDEV
# |       `-- events.out.tfevents.1649262033.amun
# |-- roi_1_metrics
# |   |-- MAE
# |   |   `-- events.out.tfevents.1649262033.amun
# |   |-- MSE
# |   |   `-- events.out.tfevents.1649262033.amun
# |   `-- PSNR
# |       `-- events.out.tfevents.1649262033.amun
# |-- roi_1_statistics
# |   |-- MEAN
# |   |   `-- events.out.tfevents.1649262033.amun
# |   `-- STDDEV
# |       `-- events.out.tfevents.1649262033.amun
# |-- roi_2_metrics
# |   |-- MAE
# |   |   `-- events.out.tfevents.1649262033.amun
# |   |-- MSE
# |   |   `-- events.out.tfevents.1649262033.amun
# |   `-- PSNR
# |       `-- events.out.tfevents.1649262033.amun
# `-- roi_2_statistics
#     |-- MEAN
#     |   `-- events.out.tfevents.1649262033.amun
#     `-- STDDEV
#         `-- events.out.tfevents.1649262033.amun
#
#
#
#
#----------------------------------------------------------------------------------------------

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# set this to the path of your experiment
exp_path = Path('runs/exp-20220406-172033')

#----------------------------------------------------------------------------------------------

# find all event files, loop over them and read them
for event_file in sorted(list(exp_path.rglob('events.out.*'))):
    print(event_file)
    if event_file.parent == exp_path:
        print('cost')
    else:
        print(event_file.parent.parent.name)
        print(event_file.parent.name)

    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    for scalar_tag in ea.Tags()['scalars']:
        #print(scalar_tag)
        df = pd.DataFrame(ea.Scalars(scalar_tag))
        print(df)

    print('')
