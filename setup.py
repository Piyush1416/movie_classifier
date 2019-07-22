from setuptools import setup

setup(name='movie_classifier',
      version='0.1',
      description='The Quickest CLI based Genre Predictor in the world',
      url='',
      author='Piyush Anasta Rumao',
      author_email='piyushrumao@gmail.com',
      license='MIT',
      packages=['movie_classifier'],
      zip_safe=False,
      install_requires = ['pandas','nltk','argparse','sklearn'],
      package_data={'movie_classifier': ['data/*.pkl']},
entry_points = {
        'console_scripts': [
            'movie_classifier = movie_classifier.__main__:main'
        ]
    }
      )
