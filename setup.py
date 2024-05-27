from distutils.core import setup
setup(
  name = 'FanFAIR',
  packages = ['fanfair'], 
  version = '1.0.1',
  description = 'FanFAIR, semi-automatic assessment of datasets fairness',
  author = 'Chiara Gallese, Marco S. Nobile',
  author_email = 'marco.nobile@unive.it',
  url = 'https://github.com/aresio/FanFAIR', # use the URL to the github repo
  keywords = ['fuzzy logic', 'dataset analysis', 'dataset fairness'], # arbitrary keywords
  license='LICENSE.txt',
  #long_description_content_type='text/markdown',
  #long_description=open('README.md', encoding='utf-8').read(),
  classifiers = ['Programming Language :: Python :: 3.7'],
  install_requires=[ 'numpy', 'simpful', 'scipy', 'pandas', 'matplotlib' ],
)
