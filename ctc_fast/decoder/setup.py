
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension("decoder", ["decoder.pyx"]),
                 Extension("clm_decoder", ["clm_decoder.pyx"]),
                 Extension("clm_decoder2", ["clm_decoder2.pyx"])]
)
