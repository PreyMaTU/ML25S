from reporting import plot_stored_crossval_scores, plot_stored_crossval_boxplots


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

  knn_heart_disease_binary= [
    ('knn', 'Heart Disease MinMax' , '5 Classes'),
    ('knn', 'Heart Disease MinMax Binary' , '2 Classes'),
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

  # plot class collapsing for heart disease dataset
  plot_stored_crossval_scores(knn_heart_disease_binary,
    score_type= 'test_accuracy',
    title= 'KNN of Heart Disease data',
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

  # plot all NN metrics over all datasets with varying learning rate
  plot_stored_crossval_scores(all_nn_minmax_learningrate,
    score_type= 'test_f1_weighted',
    title= 'NN F1 score over learning rate (Minmax)',
    xlabel= 'learning rate',
    ylabel= 'F1'
  )
  plot_stored_crossval_scores(all_nn_minmax_learningrate,
    score_type= 'test_accuracy',
    title= 'NN Accuracy over learning rate (Minmax)',
    xlabel= 'learning rate',
    ylabel= 'Accuracy'
  )
  plot_stored_crossval_scores(all_nn_minmax_learningrate,
    score_type= 'fit_time',
    title= 'NN fit time over learning rate (Minmax)',
    xlabel= 'learning rate',
    ylabel= 'fit time'
  )

    # plot all NN metrics over all datasets with varying layer structure
  plot_stored_crossval_scores(all_nn_minmax_hiddenlayers,
    score_type= 'test_f1_weighted',
    title= 'NN F1 score over hidden layers (Minmax)',
    xlabel= 'hidden layers',
    ylabel= 'F1'
  )
  plot_stored_crossval_scores(all_nn_minmax_hiddenlayers,
    score_type= 'test_accuracy',
    title= 'NN Accuracy over hidden layers (Minmax)',
    xlabel= 'hidden layers',
    ylabel= 'Accuracy'
  )
  plot_stored_crossval_scores(all_nn_minmax_hiddenlayers,
    score_type= 'fit_time',
    title= 'NN fit time over hidden layers (Minmax)',
    xlabel= 'hidden layers',
    ylabel= 'fit time'
  )

  # plot feature selection for breast cancer dataset
  plot_stored_crossval_scores(nn_breast_cancer_all_scalers,
    score_type= 'test_accuracy',
    title= 'NN of Breast Cancer data',
    xlabel= 'hidden layers',
    ylabel= 'Accuracy'
  )

  ########### RF ###########

  all_rf_scaled= [
    ('rf', 'Breast Cancer Scaled', 'Breast Cancer'),
    ('rf', 'Loan Scaled', ' Loan'),
    ('rf', 'Dota Scaled', 'Dota'),
    ('rf', 'Heart Disease Scaled', 'Heart Disease'),
  ]
  
  all_rf_depths= [
    ('rf', 'Breast Cancer Depths', 'Breast Cancer'),
    ('rf', 'Loan Depths', ' Loan'),
    ('rf', 'Dota Depths', 'Dota'),
    ('rf', 'Heart Disease Depths', 'Heart Disease'),
  ]

  rf_loan_all_scalers= [
    ('rf', 'Loan Scaled', 'Scaled'),
    ('rf', 'Loan Unscaled', 'Unscaled'),
  ]
  
  rf_dota_all_scalers= [
    ('rf', 'Dota Scaled', 'Scaled'),
    ('rf', 'Dota Unscaled', 'Unscaled'),
  ]

  rf_breast_cancer_all_scalers= [
    ('rf', 'Breast Cancer Scaled', 'Scaled'),
    ('rf', 'Breast Cancer Unscaled', 'Unscaled'),
  ]

  # plot all rf metrics over all datasets 
  plot_stored_crossval_scores(all_rf_scaled,
    score_type= 'test_f1_weighted',
    title= 'RF F1 score over #estimators (Robust Scaler)',
    xlabel= 'estimators',
    ylabel= 'F1'
  )

  plot_stored_crossval_scores(all_rf_scaled,
    score_type= 'test_accuracy',
    title= 'RF Accuracy over #estimators (Robust Scaler)',
    xlabel= 'estimators',
    ylabel= 'accuracy'
  )

  plot_stored_crossval_scores(all_rf_scaled,
    score_type= 'fit_time',
    title= 'RF fit time over #estimators (Robust Scaler)',
    xlabel= 'estimators',
    ylabel= 'fit time'
  )

  # plot accuracy over depth for all datasets
  plot_stored_crossval_scores(all_rf_depths,
    score_type= 'test_accuracy',
    title= 'RF Accuracy over depth (Robust Scaler)',
    xlabel= 'depth',
    ylabel= 'accuracy'
  )

  # plot comparison of scaled vs unscaled for loan data set
  plot_stored_crossval_scores(rf_loan_all_scalers,
    score_type= 'test_accuracy',
    title= 'RF Accuracy over #estimators (Loan)',
    xlabel= 'estimators',
    ylabel= 'accuracy'
  )

  # plot comparison of scaled vs unscaled for dota data set
  plot_stored_crossval_scores(rf_dota_all_scalers,
    score_type= 'test_accuracy',
    title= 'RF Accuracy over #estimators (Dota)',
    xlabel= 'estimators',
    ylabel= 'accuracy'
  )

  # plot comparison of scaled vs unscaled for breat cancer data set
  plot_stored_crossval_scores(rf_breast_cancer_all_scalers,
    score_type= 'test_accuracy',
    title= 'RF Accuracy over #estimators (Breast Cancer)',
    xlabel= 'estimators',
    ylabel= 'accuracy'
  )

  

  ###############  Boxplots  ##################

  #
  # TODO: Instead of comparing different configurations for one classifier,
  #       we need to compare the same dataset across differen classifieres
  #       eg like this:
  #      [
  #        ('knn', 'Breast Cancer MinMax', 'KNN MinMax'),
  #        ('rf', 'Breast Cancer Scaled', 'RF Scaled'),
  #        ('nn', 'Breast Cancer kek', 'NN kek'),
  #      ]

  plot_stored_crossval_boxplots([
      ('knn', 'Heart Disease MinMax', 'Heart Disease'),
      ('knn', 'Breast Cancer MinMax', 'Breast Cancer'),
      ('knn', 'Heart Disease MinMax Binary', 'Heart Disease 2'),
    ],
    score_type='test_f1_weighted',
    title='Comparison of F1 score distribution',
    ylabel= 'F1',
    show= True
  )
