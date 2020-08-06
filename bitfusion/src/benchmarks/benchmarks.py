import argparse
import logging

from dnnweaver2.graph import Graph, get_default_graph
from dnnweaver2.tensorOps.cnn import conv2D, maxPool, flatten, matmul, addBias, batch_norm, reorg, concat, leakyReLU, add
from dnnweaver2 import get_tensor
import logging
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

import os

def fc(tensor_in, output_channels=1024,
        f_dtype=None, w_dtype=None,
        act='linear'):
    input_channels = tensor_in.shape[-1]
    weights = get_tensor(shape=(output_channels, input_channels),
            name='weights',
            dtype=w_dtype)
    biases = get_tensor(shape=(output_channels,),
            name='biases',
            dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _fc = matmul(tensor_in, weights, biases, dtype=f_dtype)

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_fc, dtype=_fc.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _fc
    else:
        raise ValueError, 'Unknown activation type {}'.format(act)

    return act

def conv(tensor_in, filters=32, stride=None, kernel_size=3, pad='SAME',
        c_dtype=None, w_dtype=None,
        act='linear'):

    if stride is None:
        stride = (1,1,1,1)

    input_channels = tensor_in.shape[-1]

    weights = get_tensor(shape=(filters, kernel_size, kernel_size, input_channels),
                         name='weights',
                         dtype=w_dtype)
    biases = get_tensor(shape=(filters),
                         name='biases',
                         dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _conv = conv2D(tensor_in, weights, biases, stride=stride, pad=pad, dtype=c_dtype)

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_conv, dtype=_conv.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _conv
    else:
        raise ValueError, 'Unknown activation type {}'.format(act)

    return act

benchlist = [\
             'AlexNet', \
             'SVHN', \
             'CIFAR10', \
             'LeNet-5', \
             'VGG-7', \
             'RESNET-18', \
             'RNN', \
             'LSTM' \
            ]

def get_bench_nn(bench_name, WRPN=False):
    if bench_name == 'AlexNet':
        if WRPN:
            return get_alex_net_wrpn()
        else:
            return get_alex_net()
    elif bench_name == 'SVHN':
        return get_svhn_qnn()
    elif bench_name == 'CIFAR10':
        return get_cifar10_qnn()
    elif bench_name == 'LeNet-5':
        return get_lenet_5_twn()
    elif bench_name == 'VGG-7':
        return get_vgg_7_twn()
    elif bench_name == 'RESNET-18':
        if WRPN:
            return get_resnet_18_wrpn()
        else:
            return get_resnet_18_twn()
    elif bench_name == 'RESNET-20':
        return get_resnet_20_twn()
    elif bench_name == 'RNN':
        return get_RNN('RNN', 2048)
    elif bench_name == 'LSTM':
        return get_LSTM('LSTM', 900)

def write_to_csv(csv_name, fields, stats, graph, csv_path='./'):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for l in stats:
        print(l)
        print(stats[l]['total'])

    bench_csv_name = os.path.join(csv_path, csv_name)
    with open(bench_csv_name, 'w') as f:
        f.write(', '.join(fields+['\n']))
        for l in network:
            if isinstance(network[l], ConvLayer):
                f.write('{}, {}\n'.format(l, ', '.join(str(x) for x in stats[l]['total'])))

def get_bench_numbers(graph, sim_obj, batch_size=1):
    stats = {}
    for opname, op in graph.op_registry.iteritems():
        out = sim_obj.get_cycles(op, batch_size)
        if out is not None:
            s, l = out
            stats[opname] = s
    return stats

def get_alex_net():
    '''
    AlexNet
    Krizhevsky, Sutskever, and Hinton, 2012
    '''
    g = Graph('AlexNet', dataset='imagenet', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,227,227,3), name='data', dtype=FQDtype.FXP8, trainable=False)

        with g.name_scope('conv1_a'):
            conv1_a = conv(i, filters=48, kernel_size=11, stride=(1,4,4,1), pad='VALID',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool1_a'):
            pool1_a = maxPool(conv1_a, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1_b'):
            conv1_b = conv(i, filters=48, kernel_size=11, stride=(1,4,4,1), pad='VALID',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool1_b'):
            pool1_b = maxPool(conv1_b, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2_a'):
            conv2_a = conv(pool1_a, filters=128, kernel_size=5, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool2_a'):
            pool2_a = maxPool(conv2_a, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2_b'):
            conv2_b = conv(pool1_b, filters=128, kernel_size=5, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool2_b'):
            pool2_b = maxPool(conv2_b, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('concat2'):
            concat2 = concat((pool2_a, pool2_b), concat_dim=-1, dtype=FQDtype.FXP4)

        with g.name_scope('conv3_a'):
            conv3_a = conv(concat2, filters=192, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv3_b'):
            conv3_b = conv(concat2, filters=192, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv4_a'):
            conv4_a = conv(conv3_a, filters=192, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv4_b'):
            conv4_b = conv(conv3_b, filters=192, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv5_a'):
            conv5_a = conv(conv4_a, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool5_a'):
            pool5_a = maxPool(conv5_a, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('conv5_b'):
            conv5_b = conv(conv4_b, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool5_b'):
            pool5_b = maxPool(conv5_b, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('concat5'):
            concat5 = concat((pool5_a, pool5_b), concat_dim=-1)

        with g.name_scope('flatten5'):
            flatten5 = flatten(concat5)

        with g.name_scope('fc1'):
            fc1 = fc(flatten5, output_channels=4096, w_dtype=FQDtype.FXP4,
                    f_dtype=FQDtype.FXP4)

        with g.name_scope('fc2'):
            fc2 = fc(fc1, output_channels=4096, w_dtype=FQDtype.FXP4,
                    f_dtype=FQDtype.FXP8)

        with g.name_scope('fc3'):
            fc3 = fc(fc2, output_channels=1000, w_dtype=FQDtype.FXP8,
                    f_dtype=None)

    return g

def get_alex_net_wrpn():
    '''
    AlexNet
    Krizhevsky, Sutskever, and Hinton, 2012
    '''
    g = Graph('AlexNet-WRPN (2x)', dataset='imagenet', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,227,227,3), name='data', dtype=FQDtype.FXP8, trainable=False)

        with g.name_scope('conv1_a'):
            conv1_a = conv(i, filters=96, kernel_size=11, stride=(1,4,4,1), pad='VALID',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP8)
        with g.name_scope('pool1_a'):
            pool1_a = maxPool(conv1_a, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1_b'):
            conv1_b = conv(i, filters=96, kernel_size=11, stride=(1,4,4,1), pad='VALID',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP8)
        with g.name_scope('pool1_b'):
            pool1_b = maxPool(conv1_b, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2_a'):
            conv2_a = conv(pool1_a, filters=256, kernel_size=5, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool2_a'):
            pool2_a = maxPool(conv2_a, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2_b'):
            conv2_b = conv(pool1_b, filters=256, kernel_size=5, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool2_b'):
            pool2_b = maxPool(conv2_b, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('concat2'):
            concat2 = concat((pool2_a, pool2_b), concat_dim=-1, dtype=FQDtype.FXP4)

        with g.name_scope('conv3_a'):
            conv3_a = conv(concat2, filters=384, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv3_b'):
            conv3_b = conv(concat2, filters=384, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv4_a'):
            conv4_a = conv(conv3_a, filters=384, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv4_b'):
            conv4_b = conv(conv3_b, filters=384, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('conv5_a'):
            conv5_a = conv(conv4_a, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool5_a'):
            pool5_a = maxPool(conv5_a, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('conv5_b'):
            conv5_b = conv(conv4_b, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
        with g.name_scope('pool5_b'):
            pool5_b = maxPool(conv5_b, pooling_kernel=(1,2,2,1), pad='VALID')

        with g.name_scope('concat5'):
            concat5 = concat((pool5_a, pool5_b), concat_dim=-1)

        with g.name_scope('flatten5'):
            flatten5 = flatten(concat5)

        with g.name_scope('fc1'):
            fc1 = fc(flatten5, output_channels=8192, w_dtype=FQDtype.FXP4,
                    f_dtype=FQDtype.FXP4)

        with g.name_scope('fc2'):
            fc2 = fc(fc1, output_channels=8192, w_dtype=FQDtype.FXP4,
                    f_dtype=FQDtype.FXP8)

        with g.name_scope('fc3'):
            fc3 = fc(fc2, output_channels=1000, w_dtype=FQDtype.FXP8,
                    f_dtype=None)

    return g

def get_RNN(name, size):
    g = Graph(name, dataset='PTB', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,size*2), name='data', dtype=FQDtype.FXP4, trainable=False)
        with g.name_scope('matmul'):
            out = fc(i, output_channels=2*size, w_dtype=FQDtype.FXP4,
                    f_dtype=None)
    return g

def get_LSTM(name, size):
    g = Graph(name, dataset='PTB', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,size*4), name='data', dtype=FQDtype.FXP4, trainable=False)
        with g.name_scope('matmul'):
            out = fc(i, output_channels=4*size, w_dtype=FQDtype.FXP4,
                    f_dtype=None)
    return g

def get_svhn_qnn():
    '''
    SVHN
    QNN

    Weights are 1-bit
    Inputs are 8-bit for first layer, 1-bit for the rest

    '''
    g = Graph('SVHN-QNN', dataset='SVHN', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,32,32,3), name='data', dtype=FQDtype.FXP8, trainable=False)


        with g.name_scope('conv0'):
            conv0 = conv(i, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.Bin)

        with g.name_scope('conv1'):
            conv1 = conv(conv0, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.Bin)
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2'):
            conv2 = conv(pool1, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.Bin)

        with g.name_scope('conv3'):
            conv3 = conv(conv2, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.Bin)
        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv4'):
            conv4 = conv(pool3, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.Bin)

        with g.name_scope('conv5'):
            conv5 = conv(conv4, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.Bin, w_dtype=FQDtype.Bin)
        with g.name_scope('pool5'):
            pool5 = maxPool(conv5, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('flatten5'):
            flatten5 = flatten(pool5)

        with g.name_scope('fc6'):
            fc6 = fc(flatten5, output_channels=1024, w_dtype=FQDtype.Bin,
                    f_dtype=FQDtype.Bin)

        with g.name_scope('fc7'):
            fc7 = fc(fc6, output_channels=1024, w_dtype=FQDtype.Bin,
                    f_dtype=FQDtype.Bin)

        with g.name_scope('fc8'):
            fc8 = fc(fc7, output_channels=10, w_dtype=FQDtype.Bin,
                    f_dtype=None)
    return g

def get_cifar10_qnn():
    '''
    CIFAR-10
    QNN
    '''

    g = Graph('CIFAR-10-QNN', dataset='Cifar-10', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,32,32,3), name='data', dtype=FQDtype.FXP8, trainable=False)

        with g.name_scope('conv0'):
            conv0 = conv(i, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP8)

        with g.name_scope('conv1'):
            conv1 = conv(conv0, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2'):
            conv2 = conv(pool1, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)

        with g.name_scope('conv3'):
            conv3 = conv(conv2, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv4'):
            conv4 = conv(pool3, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)

        with g.name_scope('conv5'):
            conv5 = conv(conv4, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool5'):
            pool5 = maxPool(conv5, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('flatten5'):
            flatten5 = flatten(pool5)

        with g.name_scope('fc6'):
            fc6 = fc(flatten5, output_channels=1024, w_dtype=FQDtype.FXP2,
                    f_dtype=FQDtype.FXP2)

        with g.name_scope('fc7'):
            fc7 = fc(fc6, output_channels=1024, w_dtype=FQDtype.FXP2,
                    f_dtype=FQDtype.FXP2)

        with g.name_scope('fc8'):
            fc8 = fc(fc7, output_channels=10, w_dtype=FQDtype.FXP2,
                    f_dtype=None)
    return g

def get_lenet_5_twn():
    '''
    LeNet-5
    TWN
    '''

    g = Graph('LeNet-5', dataset='MNIST', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,32,32,1), name='data', dtype=FQDtype.FXP2, trainable=False)

        with g.name_scope('conv0'):
            conv0 = conv(i, filters=32, kernel_size=5, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1'):
            conv1 = conv(pool0, filters=64, kernel_size=5, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('flatten1'):
            flatten1 = flatten(pool1)

        with g.name_scope('fc1'):
            fc1 = fc(flatten1, output_channels=512, w_dtype=FQDtype.FXP2,
                    f_dtype=FQDtype.FXP2)

        with g.name_scope('fc2'):
            fc2 = fc(fc1, output_channels=10, w_dtype=FQDtype.FXP2,
                    f_dtype=FQDtype.FXP2)

    return g

def get_vgg_7_twn():
    '''
    VGG-7
    '''

    g = Graph('VGG-7-CIFAR-10', dataset='CIFAR-10', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,32,32,3), name='data', dtype=FQDtype.FXP2, trainable=False)

        with g.name_scope('conv1_a'):
            conv1_a = conv(i, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool1_a'):
            pool1_a = maxPool(conv1_a, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1_b'):
            conv1_b = conv(i, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool1_b'):
            pool1_b = maxPool(conv1_b, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2_a'):
            conv2_a = conv(pool1_a, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool2_a'):
            pool2_a = maxPool(conv2_a, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2_b'):
            conv2_b = conv(pool1_b, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool2_b'):
            pool2_b = maxPool(conv2_b, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv3_a'):
            conv3_a = conv(pool2_a, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool3_a'):
            pool3_a = maxPool(conv3_a, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv3_b'):
            conv3_b = conv(pool2_b, filters=512, kernel_size=3, pad='SAME',
                    c_dtype=FQDtype.FXP2, w_dtype=FQDtype.FXP2)
        with g.name_scope('pool3_b'):
            pool3_b = maxPool(conv3_b, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('add3'):
            add3 = add((pool3_a, pool3_b))

        with g.name_scope('flatten3'):
            flatten3 = flatten(add3)

        with g.name_scope('fc1'):
            fc1 = fc(flatten3, output_channels=1024, w_dtype=FQDtype.FXP2,
                    f_dtype=FQDtype.FXP2)

    return g

def get_resnet_18_twn():
    '''
    ResNet-18

    Note that this isn't ResNet-18B
    '''

    g = Graph('ResNet-18-TWN', dataset='ImageNet', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,224,224,3), name='data', dtype=FQDtype.FXP8, trainable=False)

        with g.name_scope('conv1_a'):
            conv1_a = conv(i, filters=64, kernel_size=7, pad='SAME', stride=(1,2,2,1),
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP8)
        with g.name_scope('pool1_a'):
            pool1_a = maxPool(conv1_a, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('module2'):
            for i in range(1, 3):
                if i == 1:
                    prev = pool1_a
                else:
                    prev = conv2_b
                with g.name_scope('conv2_{}_a'.format(i)):
                    conv2_a = conv(prev, filters=64, kernel_size=1, pad='SAME',
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv2_{}_b'.format(i)):
                    conv2_b = conv(conv2_a, filters=64, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('add2'):
            add2 = add((conv2_a, conv2_b))

        with g.name_scope('module3'):
            for i in range(1, 3):
                if i == 1:
                    prev = add2
                    stride = (1,2,2,1)
                else:
                    prev = conv3_b
                    stride = (1,1,1,1)
                with g.name_scope('conv3_{}_a'.format(i)):
                    conv3_a = conv(prev, filters=128, kernel_size=1, pad='SAME', stride=stride,
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv3_{}_b'.format(i)):
                    conv3_b = conv(conv3_a, filters=128, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('add3'):
            add3 = add((conv3_a, conv3_b))

        with g.name_scope('module4'):
            for i in range(1, 3):
                if i == 1:
                    prev = add3
                    stride = (1,2,2,1)
                else:
                    prev = conv4_b
                    stride = (1,1,1,1)
                with g.name_scope('conv4_{}_a'.format(i)):
                    conv4_a = conv(prev, filters=256, kernel_size=1, pad='SAME', stride=stride,
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv4_{}_b'.format(i)):
                    conv4_b = conv(conv4_a, filters=256, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('add4'):
            add4 = add((conv4_a, conv4_b))

        with g.name_scope('module5'):
            for i in range(1, 3):
                if i == 1:
                    prev = add4
                    stride = (1,2,2,1)
                else:
                    prev = conv5_b
                    stride = (1,1,1,1)
                with g.name_scope('conv5_{}_a'.format(i)):
                    conv5_a = conv(prev, filters=512, kernel_size=1, pad='SAME', stride=stride,
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv5_{}_b'.format(i)):
                    conv5_b = conv(conv5_a, filters=512, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
    return g

def get_resnet_18_wrpn():
    '''
    ResNet-18 according to WRPN
    '''

    g = Graph('ResNet-18-WRPN', dataset='ImageNet', log_level=logging.INFO)
    batch_size = 16

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,224,224,3), name='data', dtype=FQDtype.FXP8, trainable=False)

        with g.name_scope('conv1_a'):
            conv1_a = conv(i, filters=128, kernel_size=7, pad='SAME', stride=(1,2,2,1),
                    c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP8)
        with g.name_scope('pool1_a'):
            pool1_a = maxPool(conv1_a, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('module2'):
            for i in range(1, 3):
                if i == 1:
                    prev = pool1_a
                else:
                    prev = conv2_b
                with g.name_scope('conv2_{}_a'.format(i)):
                    conv2_a = conv(prev, filters=128, kernel_size=1, pad='SAME',
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv2_{}_b'.format(i)):
                    conv2_b = conv(conv2_a, filters=128, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('add2'):
            add2 = add((conv2_a, conv2_b))

        with g.name_scope('module3'):
            for i in range(1, 3):
                if i == 1:
                    prev = add2
                    stride = (1,2,2,1)
                else:
                    prev = conv3_b
                    stride = (1,1,1,1)
                with g.name_scope('conv3_{}_a'.format(i)):
                    conv3_a = conv(prev, filters=256, kernel_size=1, pad='SAME', stride=stride,
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv3_{}_b'.format(i)):
                    conv3_b = conv(conv3_a, filters=256, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('add3'):
            add3 = add((conv3_a, conv3_b))

        with g.name_scope('module4'):
            for i in range(1, 3):
                if i == 1:
                    prev = add3
                    stride = (1,2,2,1)
                else:
                    prev = conv4_b
                    stride = (1,1,1,1)
                with g.name_scope('conv4_{}_a'.format(i)):
                    conv4_a = conv(prev, filters=512, kernel_size=1, pad='SAME', stride=stride,
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv4_{}_b'.format(i)):
                    conv4_b = conv(conv4_a, filters=512, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)

        with g.name_scope('add4'):
            add4 = add((conv4_a, conv4_b))

        with g.name_scope('module5'):
            for i in range(1, 3):
                if i == 1:
                    prev = add4
                    stride = (1,2,2,1)
                else:
                    prev = conv5_b
                    stride = (1,1,1,1)
                with g.name_scope('conv5_{}_a'.format(i)):
                    conv5_a = conv(prev, filters=1024, kernel_size=1, pad='SAME', stride=stride,
                                c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
                with g.name_scope('conv5_{}_b'.format(i)):
                    conv5_b = conv(conv5_a, filters=1024, kernel_size=3, pad='SAME',
                            c_dtype=FQDtype.FXP4, w_dtype=FQDtype.FXP4)
    return g

if __name__ == "__main__":
    # parser object
    argp = argparse.ArgumentParser()

    # parser arguments
    argp.add_argument("-c", "--config_file", dest='config_file', default='conf.ini', type=str)
    argp.add_argument("-v", "--verbose", dest='verbose', default=False, action='store_true')

    # parse
    args = argp.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    # Read config file
    logger.info('Creating benchmarks')

    sim_obj = Simulator(args.config_file, args.verbose)
    fields = ['Layer', 'Total Cycles', 'Memory Stall Cycles', \
              'Activation Reads', 'Weight Reads', 'Output Reads', \
              'DRAM Reads', 'Output Writes', 'DRAM Writes']
    csv_dir = 'csv'
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)

    for bench in benchlist:
        print(bench)
        nn = get_bench_nn(bench)
        print(nn)
        stats = get_bench_numbers(nn, sim_obj)
        write_to_csv(os.path.join(csv_dir, bench+'.csv'), fields, stats, nn)
