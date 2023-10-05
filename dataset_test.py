from ldm.data.synthetic_data import BlockWorld
import matplotlib.pyplot as plt

dset = BlockWorld(size=None, control=['Depth', 'Env'])
ex = dset[0]
# dset = LaionValidation(size=256)
# ex = dset[0]
for k in ["image", "caption", 'control_grid'
            ]:
    print(type(ex[k]))
    try:
        print(ex[k].shape)
    except:
        print(ex[k])
    if k =='control_grid':
        plt.imshow(ex[k])
        plt.savefig('control_gird.png')