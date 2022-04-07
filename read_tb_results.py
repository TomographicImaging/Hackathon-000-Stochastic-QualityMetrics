# minimal example that shows how to read tensorboard results into pandas data frames

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

exp_path = Path('runs/exp-20220406-172033')

for event_file in sorted(list(exp_path.rglob('events.out.*'))):
    #print(event_file)
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
