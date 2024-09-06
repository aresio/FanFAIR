from distutils.core import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
  name = 'FanFAIR',
  packages = ['fanfair'], 
  version = '1.0.2',
  description = 'FanFAIR, semi-automatic assessment of datasets fairness',
  author = 'Chiara Gallese, Marco S. Nobile',
  author_email = 'marco.nobile@unive.it',
  url = 'https://github.com/aresio/FanFAIR', # use the URL to the github repo
  keywords = ['fuzzy logic', 'dataset analysis', 'dataset fairness'], # arbitrary keywords
  license='LICENSE.txt',
  long_description=long_description,
  long_description_content_type='text/markdown',
  classifiers = ['Programming Language :: Python :: 3.7'],
  install_requires=[ 'numpy', 'simpful', 'scipy', 'pandas', 'matplotlib' ],
)
