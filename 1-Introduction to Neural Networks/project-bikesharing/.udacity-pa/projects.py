import argparse
import subprocess as sp
from udacity_pa import udacity

nanodegree = 'nd101'
projects = ['first_neural_network']
filenames = ['my_answers.py', 'Predicting_bike_sharing_data.ipynb']

def submit(args):

  # Do we prefer ipynb, html or both?
  # sp.call(['jupyter', 'nbconvert', '--to', 'html', 'dlnd_image_classification.ipynb'])

  udacity.submit(nanodegree, projects[0], filenames, 
                 environment = args.environment,
                 jwt_path = args.jwt_path)
