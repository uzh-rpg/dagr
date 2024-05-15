from distutils.core import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dagr',
    packages=['dagr'],
    package_dir={'':'src'},
    ext_modules=[
        CUDAExtension(name='asy_tools',
                      sources=['src/dagr/asynchronous/asy_tools/main.cu']),
        CUDAExtension(name="ev_graph_cuda",
                      sources=['src/dagr/graph/ev_graph.cu'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
