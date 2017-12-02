import pytest

from resnext import ResNeXt
from resnext import Conv2DGroup

from keras import backend as K
from keras.layers import Input

def test_resnext29():
    net = ResNeXt(input_shape=[32, 32, 3], num_class=10, cardinality=8, depth=29, 
                        base_width=64)
    model = net.build()
    assert model.output_shape == (None, 10)

def test_conv2dgroup():
    inputs = Input(shape=[8, 8, 32])
    ouputs = Conv2DGroup(filters=64, kernel_size=3, groups=2, strides=1, padding='same',
                        use_bias=False)(inputs)
    assert K.int_shape(ouputs) == (None, 8, 8, 64)

if __name__ == '__main__':
    
    pytest.main([__file__])