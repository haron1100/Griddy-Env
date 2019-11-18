from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()
    
setup(
    name = 'GriddyEnv',
    version = '0.0.1',
    description = 'Grid Environment for testing and teaching RL algorithms. Integrated into OpenAI Gym.',   # Give a short description about your library
    long_description = long_description,
    long_description_content_type='text/markdown',
    author = 'Haron Shams',                   # Type in your name
    author_email = 'hshams@hotmail.co.uk',      # Type in your E-Mail
    url = 'https://github.com/haron1100/Griddy-Env',   # Provide either the link to your github or to your website
    py_modules=["__init__"],
    package_dir={'':'GriddyEnv'},
    classifiers=[
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Operating System :: OS Independent',
  ],
)


'''from distutils.core import setup
setup(
  name = 'GriddyEnv',         # How you named your package folder (MyLib)
  packages = ['GriddyEnv'],   # Chose the same as "name"
  version = '0.0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Grid Environment for testing and teaching RL algorithms. Integrated into OpenAI Gym.',   # Give a short description about your library
  download_url = 'https://github.com/haron1100/Griddy-Env/archive/v0.0.1.tar.gz',    # I explain this later on
  keywords = ['Reinforcement Learning', 'AI', 'ML', 'Machine Learning', 'Artificial Intelligence', 'Simulation'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'gym',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
'''
