from reporting import plot_stored_crossval_scores, plot_stored_crossval_boxplots


def plotting():
  
  ########### KNN ###########
  all_knn_minmax= [
      ('knn', 'Breast Cancer MinMax', 'Breast Cancer'),
      ('knn', 'Loan MinMax 10 Features', ' Loan'),
      ('knn', 'Dota MinMax', 'Dota'),
      ('knn', 'Heart Disease MinMax', 'Heart Disease'),
      ('knn', 'Heart Disease MinMax Binary' , 'Heart Disease Binary'),
    ]
  
  knn_breast_cancer_all_scales= [
    ('knn', 'Breast Cancer MinMax', 'MinMax'),
    ('knn', 'Breast Cancer Standard', 'Standard'),
    ('knn', 'Breast Cancer No Scale', 'No Scale'),
  ]

  knn_loan_all_scales= [
    ('knn', 'Loan MinMax 10 Features', 'MinMax'),
    ('knn', 'Loan Standard 10 Features', 'Standard'),
    ('knn', 'Loan No Scale 10 Features', 'No Scale'),
  ]

  knn_dota_all_scales= [
    ('knn', 'Dota No scale', 'No Scale'),
    ('knn', 'Dota MinMax', 'MinMax'),
  ]

  knn_heart_disease_all_scales= [
    ('knn', 'Heart Disease MinMax', 'MinMax'),
    ('knn', 'Heart Disease Standard', 'Standard'),
    ('knn', 'Heart Disease No Scale', 'No Scale'),
  ]

  knn_loan_feature_selection= [
    ('knn', 'Loan MinMax All Features', 'All'),
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
    ylabel= 'fit time [s]'
  )

  # plot all scaling methods for knn for Breast Cancer dataset
  plot_stored_crossval_scores(knn_breast_cancer_all_scales,
    score_type= 'test_accuracy',
    title= 'KNN of Breast Cancer data',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )
  plot_stored_crossval_scores(knn_breast_cancer_all_scales,
    score_type= 'test_f1_weighted',
    title= 'KNN of Breast Cancer data',
    xlabel= 'k',
    ylabel= 'F1'
  )

  # plot all scaling methods for knn for loan dataset
  plot_stored_crossval_scores(knn_loan_all_scales,
    score_type= 'test_accuracy',
    title= 'KNN of Loan data',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )
  plot_stored_crossval_scores(knn_loan_all_scales,
    score_type= 'test_f1_weighted',
    title= 'KNN of Loan data',
    xlabel= 'k',
    ylabel= 'F1'
  )

  # plot all scaling methods for knn for dota dataset
  plot_stored_crossval_scores(knn_dota_all_scales,
    score_type= 'test_accuracy',
    title= 'KNN of Dota data',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )
  plot_stored_crossval_scores(knn_dota_all_scales,
    score_type= 'test_f1_weighted',
    title= 'KNN of Dota data',
    xlabel= 'k',
    ylabel= 'F1'
  )

  # plot all scaling methods for knn for heart disease dataset
  plot_stored_crossval_scores(knn_heart_disease_all_scales,
    score_type= 'test_accuracy',
    title= 'KNN of Heart Disease data',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )
  plot_stored_crossval_scores(knn_heart_disease_all_scales,
    score_type= 'test_f1_weighted',
    title= 'KNN of Heart Disease data',
    xlabel= 'k',
    ylabel= 'F1'
  )

  # plot feature selection for loan dataset
  plot_stored_crossval_scores(knn_loan_feature_selection,
    score_type= 'test_accuracy',
    title= 'KNN of Loan data (Features)',
    xlabel= 'k',
    ylabel= 'Accuracy'
  )

  


  ########### NN ###########
  all_nn_minmax_learningrate= [
    ('NN', 'Breast Cancer Learning Rate MinMax', 'Breast Cancer'),
    ('NN', 'Loan Learning Rate MinMax', 'Loan'),
    ('NN', 'Dota Learning Rate MinMax', 'Dota'),
    ('NN', 'Heart Disease Learning Rate MinMax', 'Heart Disease'),
    ('NN', 'Heart Disease Binary Learning Rate MinMax', 'Heart Disease Binary')
  ]

  all_nn_minmax_hiddenlayers= [
    ('NN', 'Breast Cancer Layer Sizes MinMax', 'Breast Cancer'),
    ('NN', 'Loan Layer Sizes MinMax', 'Loan'),
    ('NN', 'Dota Layer Sizes MinMax', 'Dota'),
    ('NN', 'Heart Disease Layer Sizes MinMax', 'Heart Disease'),
    ('NN', 'Heart Disease Binary Layer Sizes MinMax', 'Heart Disease Binary')

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
    ylabel= 'F1',
    logarithmicX= True	
  )
  plot_stored_crossval_scores(all_nn_minmax_learningrate,
    score_type= 'test_accuracy',
    title= 'NN Accuracy over learning rate (Minmax)',
    xlabel= 'learning rate',
    ylabel= 'Accuracy',
    logarithmicX= True	
  )
  plot_stored_crossval_scores(all_nn_minmax_learningrate,
    score_type= 'fit_time',
    title= 'NN fit time over learning rate (Minmax)',
    xlabel= 'learning rate',
    ylabel= 'fit time [s]',
    logarithmicX= True	
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
    ylabel= 'fit time [s]'
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
    ('rf', 'Heart Disease Binary Scaled', 'Heart Disease Binary'),
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

  rf_heart_disease_all_scalers= [
    ('rf', 'Heart Disease Scaled', 'Scaled'),
    ('rf', 'Heart Disease Unscaled', 'Unscaled'),
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
    ylabel= 'fit time [s]'
  )

  # plot comparisons over various depths for all datasets
  # Accuracy
  plot_stored_crossval_scores(all_rf_depths,
    score_type= 'test_accuracy',
    title= 'RF Accuracy over depth (Robust Scaler)',
    xlabel= 'depth',
    ylabel= 'accuracy'
  )

  # F1
  plot_stored_crossval_scores(all_rf_depths,
    score_type= 'test_f1_weighted',
    title= 'RF F1 score over depth (Robust Scaler)',
    xlabel= 'depth',
    ylabel= 'F1'
  )

  # Fit time
  plot_stored_crossval_scores(all_rf_depths,
    score_type= 'fit_time',
    title= 'RF fit time over depth (Robust Scaler)',
    xlabel= 'depth',
    ylabel= 'fit time [s]'
  )

  # plot comparison of scaled vs unscaled
  # Loan
  plot_stored_crossval_scores(rf_loan_all_scalers,
    score_type= 'test_f1_weighted',
    title= 'RF F1 score over #estimators (Loan)',
    xlabel= 'estimators',
    ylabel= 'F1'
  )

  # Dota
  plot_stored_crossval_scores(rf_dota_all_scalers,
    score_type= 'test_f1_weighted',
    title= 'RF F1 score over #estimators (Dota)',
    xlabel= 'estimators',
    ylabel= 'F1'
  )

  # Breast cancer
  plot_stored_crossval_scores(rf_breast_cancer_all_scalers,
    score_type= 'test_f1_weighted',
    title= 'RF F1 score over #estimators (Breast Cancer)',
    xlabel= 'estimators',
    ylabel= 'F1'
  )

  # Heart Disease
  plot_stored_crossval_scores(rf_heart_disease_all_scalers,
    score_type= 'test_f1_weighted',
    title= 'RF F1 score over #estimators (Heart Disease)',
    xlabel= 'estimators',
    ylabel= 'F1'
  )

  

  

  ###############  Boxplots  ##################

  classifier_boxplot_breast_cancer= [
    ('knn', 'Breast Cancer MinMax', 'KNN'),
    ('rf', 'Breast Cancer Scaled', 'RF'),
    ('NN', 'Breast Cancer Layer Sizes MinMax', 'NN'),
  ]

  classifier_boxplot_loan= [
    ('knn', 'Loan MinMax 10 Features', 'KNN'),
    ('rf', 'Loan Scaled', 'RF'),
    ('NN', 'Loan Layer Sizes MinMax', 'NN'),
  ]

  classifier_boxplot_dota= [
    ('knn', 'Dota MinMax', 'KNN'),
    ('rf', 'Dota Scaled', 'RF'),
    ('NN', 'Dota Layer Sizes MinMax', 'NN'),
  ]
  
  classifier_boxplot_heart_disease= [
    ('knn', 'Heart Disease MinMax', 'KNN'),
    ('rf', 'Heart Disease Scaled', 'RF'),
    ('NN', 'Heart Disease Layer Sizes MinMax', 'NN'),
  ]
  
  # Compare F1 scores
  plot_stored_crossval_boxplots( classifier_boxplot_breast_cancer,
    score_type='test_f1_weighted',
    title='Comparison of F1 score distribution on Breast Cancer',
    ylabel= 'F1',
    show= False
  )


  plot_stored_crossval_boxplots( classifier_boxplot_loan,
    score_type='test_f1_weighted',
    title='Comparison of F1 score distribution on Loan',
    ylabel= 'F1',
    show= False
  )
  
  plot_stored_crossval_boxplots(classifier_boxplot_dota,
    score_type='test_f1_weighted',
    title='Comparison of F1 score distribution on Dota',
    ylabel= 'F1',
    show= False
  )

  plot_stored_crossval_boxplots(classifier_boxplot_heart_disease,
    score_type='test_f1_weighted',
    title='Comparison of F1 score distribution Heart Disease',
    ylabel= 'F1',
    show= False
  )

  # Compare fit times

  plot_stored_crossval_boxplots( classifier_boxplot_breast_cancer,
    score_type='fit_time',
    title='Comparison of fit time distribution on Breast Cancer',
    ylabel= 'fit time [s]',
    show= False
  )

  plot_stored_crossval_boxplots( classifier_boxplot_loan,
    score_type='fit_time',
    title='Comparison of fit time distribution on Loan',
    ylabel= 'fit time [s]',
    show= False
  )
  
  plot_stored_crossval_boxplots(classifier_boxplot_dota,
    score_type='fit_time',
    title='Comparison of fit time distribution on Dota',
    ylabel= 'fit time [s]',
    show= False
  )

  plot_stored_crossval_boxplots(classifier_boxplot_heart_disease,
    score_type='fit_time',
    title='Comparison of fit time distribution Heart Disease',
    ylabel= 'fit time [s]',
    show= False
  )
