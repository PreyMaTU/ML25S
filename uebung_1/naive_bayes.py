from reporting import classifier_header
from dataset_dota import encode_dataset_dota
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

header= classifier_header('NB')


def dataset_dota_naive_bayes_default( x, y ):
  header()
  
  x, y= encode_dataset_dota( x, y )

  alphas = np.logspace(-4, 1, 10)

  for alpha in alphas:
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('nb', BernoulliNB(alpha=alpha))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
    
    print( f'\nAlpha: {alpha}' )

    for metric in ['test_accuracy', 'test_f1_weighted', 'fit_time']:
      print(f"{metric}: {cv_scores[metric].mean():.4f}")
