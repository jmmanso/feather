from setuptools import setup

setup(name='feather',
      version='1.0',
      description='Light-weight deep learning: MLPs and stacked autoencoders',
      keywords='autoencoders, multilayer perceptron',
      url='https://github.kdc.capitalone.com/ezw112/feather',
      author='Jesus Martinez-Manso',
      author_email='jesus.martinezmanso@capitalone.com',
      install_requires = ['numpy'],
      packages=['feather'],
      zip_safe=False)
