import mmpose
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

try:
    print('mmpose version:', mmpose.__version__)
    print('cuda version:', get_compiling_cuda_version())
    print('compiler information:', get_compiler_version())
except:
    print('environment error')
finally:
    print('ALL Clear🎉🎊')

