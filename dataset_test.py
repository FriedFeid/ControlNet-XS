from ldm.data.synthetic_data import BlockWorld
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

dset = BlockWorld(
                 size=None, random_resized_crop=False,
                 interpolation='bicubic', scale=[1.0, 1.0], control=['Depth', 'Env', 'SSphere'],
                 full_set=False, data_start=50_000, data_stop=50_400,
                 data_root=None,
                 use_pillow=True
                  )
ex = dset[0]
# dset = LaionValidation(size=256)
# ex = dset[0]

for k in ["image", "caption", 'control_grid']:
    print(type(ex[k]))
    try:
        print(ex[k].shape)
    except:
        print(ex[k])
    if k == 'control_grid':
        plt.imshow(ex[k])
        plt.savefig('control_gird.png')

train_dataloader = DataLoader(dset, batch_size=8, shuffle=True)

train_batch = next(iter(train_dataloader))
for k in ["image", "caption", 'control_grid']:
    print(type(train_batch[k]))
    try:
        print(train_batch[k].shape)
    except:
        print(train_batch[k])
