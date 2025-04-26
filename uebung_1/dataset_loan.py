from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd

def dataset_loan_numeric_distributed_columns():
  return [
    'loan_amnt',
    'funded_amnt',
    'funded_amnt_inv',
    'int_rate',
    'installment',
    'annual_inc',
    'dti',
    'delinq_2yrs',
    'fico_range_low',
    'fico_range_high',
    'inq_last_6mths',
    'open_acc',
    'pub_rec',
    'revol_bal',
    'revol_util',
    'total_acc',
    'out_prncp',
    'out_prncp_inv',
    'total_pymnt',
    'total_pymnt_inv',
    'total_rec_prncp',
    'total_rec_int',
    'total_rec_late_fee',
    'recoveries',
    'collection_recovery_fee',
    'last_pymnt_amnt',
    'last_fico_range_high',
    'last_fico_range_low',
    'collections_12_mths_ex_med',
    'policy_code',
    'acc_now_delinq',
    'tot_coll_amt',
    'tot_cur_bal',
    'total_rev_hi_lim',
    'acc_open_past_24mths',
    'avg_cur_bal',
    'bc_open_to_buy',
    'bc_util',
    'chargeoff_within_12_mths',
    'delinq_amnt',
    'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl',
    'mort_acc',
    'mths_since_recent_bc',
    'mths_since_recent_inq',
    'num_accts_ever_120_pd',
    'num_actv_bc_tl',
    'num_actv_rev_tl',
    'num_bc_sats',
    'num_bc_tl',
    'num_il_tl',
    'num_op_rev_tl',
    'num_rev_accts',
    'num_rev_tl_bal_gt_0',
    'num_sats',
    'num_tl_120dpd_2m',
    'num_tl_30dpd',
    'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m',
    'pct_tl_nvr_dlq',
    'percent_bc_gt_75',
    'pub_rec_bankruptcies',
    'tax_liens',
    'tot_hi_cred_lim',
    'total_bal_ex_mort',
    'total_bc_limit',
    'total_il_high_credit_limit'
  ]


def dataset_loan_numeric_ordinal_columns():
  return [
    'issue_d',
    'earliest_cr_line',
    'last_pymnt_d',
    'last_credit_pull_d',
    'emp_length'
  ]


def encode_dataset_loan( x: pd.DataFrame, y: pd.Series ):
  if y is not None:
    y = y.to_frame(name='label')
    y = pd.Categorical( y.label ).codes  # convert classes A-G to numbers

  x= x.copy()
  
  # Binary categories
  x['term_60months'] = x.term.map( lambda t : 1 if t.strip() == '60 months' else 0 )
  x['pymnt_plan'] = x.pymnt_plan.map( lambda t : 1 if t.strip().lower() == 'y' else 0 )
  x['hardship_flag'] = x.hardship_flag.map( lambda t : 1 if t.strip().lower() == 'y' else 0 )
  x['debt_settlement_flag'] = x.debt_settlement_flag.map( lambda t : 1 if t.strip().lower() == 'y' else 0 )
  x['initial_list_status_whole'] = x.initial_list_status.map( lambda t : 1 if t.strip().lower() == 'w' else 0 )
  x['application_type_individual'] = x.application_type.map( lambda t : 1 if t.strip() == 'Individual' else 0 )
  x['disbursement_method_cash'] = x.disbursement_method.map( lambda t : 1 if t.strip() == 'Cash' else 0 )

  # Drop all the binary columns that were renamed
  x = x.drop([
    'term', 'application_type', 'disbursement_method', 'initial_list_status'
  ], axis=1)

  # Ordinal categories
  emp_length_mapping = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
  }
  x.emp_length = x.emp_length.map(emp_length_mapping).astype(int)

  # One-Hot encoded
  x = pd.get_dummies(x, columns=[
    'home_ownership', 'loan_status', 'verification_status', 'purpose', 'addr_state'
  ])
  

  # Combine _year + _month columns into a single float

  x['issue_d'] = x.issue_d_year + ( x.issue_d_month / 12.0 )
  x['earliest_cr_line'] = x.earliest_cr_line_year + ( x.earliest_cr_line_month / 12.0 )
  x['last_pymnt_d'] = x.last_pymnt_d_year + ( x.last_pymnt_d_month / 12.0 )
  x['last_credit_pull_d'] = x.last_credit_pull_d_year + ( x.last_credit_pull_d_month / 12.0 )

  x= x.drop([
    'issue_d_month', 'issue_d_year', 'earliest_cr_line_month', 'earliest_cr_line_year',
    'last_pymnt_d_month', 'last_pymnt_d_year', 'last_credit_pull_d_month', 'last_credit_pull_d_year'
  ], axis=1)

  return x, y

