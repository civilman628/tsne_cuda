from setuptools import setup

setup(name='t_sne_bhcuda',
      version='0.1',
      description='T-sne with burnes hut and cuda extension (with python wrappers also for spike sorting)',
      url='https://github.com/georgedimitriadis/t_sne_bhcuda',
      author='George Dimitriadis',
      author_email='gdimitri@hotmail.com',
      license='MIT',
      packages=['t_sne_bhcuda'],
      install_requires=[
          'markdown',
      ],
      zip_safe=False)