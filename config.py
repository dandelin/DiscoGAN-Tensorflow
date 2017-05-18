from math import ceil

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def config(name, shape):
    conf = adict()
    conf.w = shape.width
    conf.h = shape.height
    conf.c = shape.channel
    conf.batch_size = 8
    conf.test_root = name
    conf.log_dir = name + '/logs'
    conf.checkpoint_dir = name + '/checkpoint'
    conf.snapshot_dir = name + '/shapshots'
    layer_depth = 2

    conf.gen_conv_infos = {
        "conv_layer_number": layer_depth,
        "filter":[
            [4, 4, conf.c, 64],
            [4, 4, 64, 64*2],
            #[4, 4, 64*2, 64*4],
            #[4, 4, 64*4, 64*8],
        ],
        "stride" : [[1, 2, 2, 1] for _ in range(layer_depth)],
    }

    conf.gen_deconv_infos = {
        "conv_layer_number": layer_depth,
        "filter":[
            #[4, 4, 64*4, 64*8],
            #[4, 4, 64*2, 64*4],
            [4, 4, 64*1, 64*2],
            [4, 4, conf.c, 64],
        ],
        "stride" : [[1, 2, 2, 1] for _ in range(layer_depth)],
        "output_dims" : [
            #[conf.batch_size, ceil(conf.h/8), ceil(conf.w/8), 64*4],
            #[conf.batch_size, ceil(conf.h/4), ceil(conf.w/4), 64*2],
            [conf.batch_size, ceil(conf.h/2), ceil(conf.w/2), 64*1],
            [conf.batch_size, conf.h, conf.w, conf.c]
        ],
    }

    conf.disc_conv_infos = {
        "conv_layer_number": layer_depth + 1,
        "filter":[
            [4, 4, conf.c, 64],
            [4, 4, 64, 64*2],
            #[4, 4, 64*2, 64*4],
            #[4, 4, 64*4, 64*8],
            [4, 4, 64*2, 1],
        ],
        "stride" : [[1, 2, 2, 1] for _ in range(layer_depth + 1)],
    }

    return conf