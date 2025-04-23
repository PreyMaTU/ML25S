from reporting import plot_stored_crossval_scores


def plotting():
  
  ########### KNN ###########
  all_knn_minmax= [
      ('knn', 'Breast Cancer MinMax', 'Breast Cancer'),
      ('knn', 'Loan MinMax 10 Features', ' Loan'),
      ('knn', 'Dota MinMax', 'Dota'),
      ('knn', 'Heart Disease MinMax', 'Heart Disease'),
    ]
  
  knn_breast_cancer_all_scales= [
    ('knn', 'Breast Cancer MinMax', 'MinMax'),
    ('knn', 'Breast Cancer Standard', 'Standard'),
    ('knn', 'Breast Cancer No Scale', 'No Scale'),
  ]

  knn_loan_feature_selection= [
    ('knn', 'Loan MinMax 10 Features', '10 Features'),
    ('knn', 'Loan MinMax 1 Feature', '1 Feature'),
  ]

  # plot all knn metrics over all datasets 
  plot_stored_crossval_scores(all_knn_minmax,
    score_type= 'test_f1_weighted',
    title= 'KNN F1 score over k (Minmax)',
    xlabel= 'k',
    ylabel= 'F1'
  )
  plot_stored_crossval_scores(all_knn_minmax,
    score_type= 'test_accuracy',
    title= 'KNN Accuracy over k (Minmax)',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )
  plot_stored_crossval_scores(all_knn_minmax,
    score_type= 'fit_time',
    title= 'KNN fit time over k (Minmax)',
    xlabel= 'k',
    ylabel= 'fit time'
  )

  # plot all scaling methods for knn for Breast Cancer dataset
  plot_stored_crossval_scores(knn_breast_cancer_all_scales,
    score_type= 'test_accuracy',
    title= 'KNN of Breast Cancer data',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )

  # plot feature selection for loan dataset
  plot_stored_crossval_scores(knn_loan_feature_selection,
    score_type= 'test_accuracy',
    title= 'KNN of Loan data',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )


  ########### NN ###########
  all_nn_minmax_learningrate= [
    ('NN', 'Breast Cancer Learning Rate MinMax', 'Breast Cancer'),
    ('NN', 'Loan Learning Rate MinMax', 'Loan'),
    ('NN', 'Dota Learning Rate MinMax', 'Dota'),
    ('NN', 'Heart Disease Learning Rate MinMax', 'Heart Disease')
  ]

  all_nn_minmax_hiddenlayers= [
    ('NN', 'Breast Cancer Layer Sizes MinMax', 'Breast Cancer'),
    ('NN', 'Loan Layer Sizes MinMax', 'Loan'),
    ('NN', 'Dota Layer Sizes MinMax', 'Dota'),
    ('NN', 'Heart Disease Layer Sizes MinMax', 'Heart Disease')

  ]

  nn_breast_cancer_all_scalers = [
    ('NN', 'Breast Cancer Layer Sizes MinMax', 'MinMax'),
    ('NN', 'Breast Cancer Layer Sizes Standard', 'Standard'),
    ('NN', 'Breast Cancer Layer Sizes Robust', 'Robust'),
    ('NN', 'Breast Cancer Layer Sizes No Scale', 'No Scale')
  ]

  ########### RF ###########
