import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
import xgboost as xgb

np.random.seed(2018)

trn = pd.read_csv('./input/train_ver2.csv')
tst = pd.read_csv('./input/test_ver2.csv')

# 제품 변수를 별도로 저장
prods = trn.columns[24:].tolist()

# 제품 변수 결측값을 미리 0으로 대체
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

# 24개 제품 중 하나도 보유하지 않은 고객 데이터 제거
no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]
# 훈련 데이터와 테스트 데이터 통합
# 테스트 데이터에 없는 제품 변수는 0으로 채움
for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn, tst], axis=0)
# 학습에 사용할 변수를 담는 list
features = []

# 범주형 변수를 .factorize() 함수를 통해 label encoding한다.
categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo',
                   'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada',
                   'indfall', 'tipodom', 'nomprov', 'segmento']
for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)

features += categorical_cols
# 수치형 변수의 특이값과 결측값을 -99로 대체하고, 정수형으로 변환한다.
df['age'].replace(' NA', -99, inplace=True)
df['age'] = df['age'].astype(np.int8)

df['antiguedad'].replace('     NA', -99, inplace=True)
df['antiguedad'] = df['antiguedad'].astype(np.int8)

df['renta'].replace('         NA', -99, inplace=True)
df['renta'].fillna(-99, inplace=True)
df['renta'] = df['renta'].astype(float).astype(np.int8)

df['indrel_1mes'].replace('P', 5, inplace=True)
df['indrel_1mes'].fillna(-99, inplace=True)
df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)


# 학습에 사용할 수치형 변수를 features에 추가한다.
features += ['age', 'antiguedad', 'renta', 'ind_nuevo', 'indrel', 'indrel_1mes', 'ind_actividad_cliente']
# 두 날짜 변수에서 연도와 월 정보를 추출
df['fecha_alta_month'] = df['fecha_alta'].map(
    lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(
    lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int8)

features += ['fecha_alta_month', 'fecha_alta_year']
# 1등급 고객이였던 마지막 날짜
df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(
    lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(
    lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int8)

features += ['ult_fec_cli_1t_month', 'ult_fec_cli_1t_year']
# 그 외 변수의 결측값은 모두 -99로 대체
df.fillna(-99, inplace=True)
# lag-1 데이터 생성

# 날짜를 숫자로 변환하는 함수
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split('-')]
    int_date = (Y - 2015) * 12 + M
    return int_date

# 날짜를 숫자로 변환하여 int_date에 저장
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

# 데이터를 복사하고, int_date 날짜에 1을 더하여 lag을 생성
# 변수명에 _prev를 추가
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col
                  for col in df.columns]
df_lag['int_date'] += 1

# 원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다.
df_trn = df.merge(df_lag, on=['ncodpers', 'int_date'], how='left')

# 메모리 효율을 위해 불필요한 변수 삭제
del df, df_lag

# 저번 달의 제품 정보가 존재하지 않을 경우 0으로 대체
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)
df_trn.fillna(-99, inplace=True)

# lag-1 변수 추가
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]

use_dates = ['2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']
trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]
tst = df_trn[df_trn['fecha_dato'] == '2016-06-28']
del df_trn

# 훈련 데이터에서 신규 구매 건수만 추출
X = []
Y = []
for i, prod in enumerate(prods):
    prev = prod + '_prev'
    prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]
    prY = np.zeros(prX.shape[0], dtype=np.int8) + i
    X.append(prX)
    Y.append(prY)
# 신규 구매 건수
XY = pd.concat(X)
# 어떤 상품을 신규 구매했는지 인덱스로 저장
Y = np.hstack(Y)
XY['y'] = Y

# 훈련 검증 데이터로 분리
vld_date = '2016-05-28'
XY_trn = XY[XY['fecha_dato'] != vld_date]
XY_vld = XY[XY['fecha_dato'] == vld_date]

# 훈련, 검증 데이터를 XGBoost 형태로 변환한다.
X_trn = XY_trn.as_matrix(columns=features)
Y_trn = XY_trn.as_matrix(columns=['y'])
dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

X_vld = XY_vld.as_matrix(columns=features)
Y_vld = XY_vld.as_matrix(columns=['y'])
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

import pickle
# 모델 로드(자꾸 커널이 죽어서 시간 절약을 위해 로드)
with open('./model/xgb.baseline.pkl', 'rb') as f:
    model = pickle.load(f)
    best_ntree_limit = model.best_ntree_limit

from mapk import mapk

# 검증 데이터에 대한 MAP@7 값을 구하는 코드
# MAP@7 평가 척도를 위한 준비 작업이다.
# 고객 식별 번호를 추출한다.
vld = trn[trn['fecha_dato'] == vld_date]
ncodpers_vld = vld.as_matrix(columns=['ncodpers'])
# 검증 데이터에서 신규 구매를 구한다.
for prod in prods:
    prev = prod + '_prev'
    padd = prod + '_add'
    vld[padd] = vld[prod] - vld[prev]

add_vld = vld.as_matrix(columns=[prod + '_add' for prod in prods])
add_vld_list = [list() for i in range(len(ncodpers_vld))]

# 고객별 신규 구매 정답값을 add_vld_list에 저장하고, 
# 총 count를 count_vld에 저장한다.
count_vld = 0
for ncodper in range(len(ncodpers_vld)):
    for prod in range(len(prods)):
        if add_vld[ncodper, prod] > 0:
            add_vld_list[ncodper].append(prod)
            count_vld += 1

# 검증 데이터에서 얻을 수 있는 MAP@7 최고점을 미리 구한다.
# 0.04266379915553903
print(mapk(add_vld_list, add_vld_list, 7, 0.0))

# 검증 데이터에 대한 예측 값을 구한다
X_vld = vld.as_matrix(columns = features)
Y_vld = vld.as_matrix(columns=['y'])
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)
preds_vld = model.predict(dvld, ntree_limit=best_ntree_limit)

# 저번 달에 보유한 제품은 신규 구매가 불가하기 때문에, 확률값에서 미리 1을 빼준다.
preds_vld = preds_vld - vld.as_matrix(columns=[prod + '_prev' for prod in prods])

# 검증 데이터 예측 상위 7개를 추출한다.
result_vld = []
for ncodper, pred in zip(ncodpers_vld, preds_vld):
    print(ncodper)
    y_prods = [(y, p, ip) for y, p, ip in zip(pred, prods, range(len(prods)))]
    y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
    result_vld.append([ip for y, p, ip in y_prods])

# 검증 데이터에서의 MAP@7점수를 구한다.
print(mapk(add_vld_list, result_vld, 7, 0.0))
