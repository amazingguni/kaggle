import math
import io

import gzip
import pickle
import zlib

import pandas as pd
import numpy as np

# 범주형 데이터를 수치형으로 변환
from sklearn.preprocessing import LabelEncoder

# import engines
from santander_product_recommendation.winner.utils import\
    products, dtypes, date_to_float, date_to_int, mapk

from santander_product_recommendation.winner import engines

np.random.seed(2019)
transformers = {}

# onehot encoding을 구현
# e.g, 3 --> 0,0,0,1
def custom_one_hot(df, features, name, names, dtype=np.int8, check=False):
    for n, val in names.items():
        # 신규 변수명을 "변수명_숫자"로 지정한다
        new_name = f'{name}_{n}'
        # 기존 변수에서 해당 고유값을 가지면 1, 그 외는 0인 이진 변수를 생성
        df[new_name] = df[name].map(lambda x: 1 if x == val else 0).astype(dtype)
        features.append(new_name)


def label_encode(df, features, name):
    # df의 변수 name의 값을 모두 string으로 변환
    df[name] = df[name].astype('str')
    # 이미 label_encode 했던 변수는 transformer[name] 재활용
    if name in transformers:
        df[name] = transformers[name].transform(df[name])
    else:
        transformers[name] = LabelEncoder()
        df[name] = transformers[name].fit_transform(df[name])
    # label encoding한 변수를 features 리스트에 추가
    features.append(name)

# pd.Series에서 빈도가 가장 높은 100개의 고유값을 순위로 대체하고 나머지는 0으로 변환
# 고빈도 데이터에 집중하기 위함
def encode_top(s, count=100, dtype=np.int8):
    # 모든 고유값에 대한 빈도 계산
    uniqs, freqs = np.unique(s, return_counts=True)
    # 빈도 Top 100을 추출
    top = sorted(zip(uniqs, freqs), key=lambda vk: vk[1], reverse=True)[:count]
    # { 기존데이터: 순위 } 를 나타내는 dict() 생성
    top_map = { uf[0]: l+1 for uf, l in zip(top, range(len(top)))}
    # 고빈도 100개의 데이터는 순위로 대체하고 그 외는 0으로 대체
    return s.map(lambda x: top_map.get(x, 0)).astype(dtype)


# apply_transform에서는 
# (1)결측값 대체, (2)범주형 데이터 label encoding, (3)고빈도 top 100개를 빈도 순위로 변환
# (4)수치형 변수 log transformation (5)날짜 데이터에서 년/월 추출 (6)날짜 데이터 간의 차이값으로 파생변수 생성
# (7)one-got-encoding 변수 생성
def apply_transforms(train_df):
    # 학습에 사용할 변수를 저장할 features 리스트 생성
    features = []
    
    # 두 변수를 label_encode()
    label_encode(train_df, features, 'canal_entrada')
    label_encode(train_df, features, 'pais_residencia')

    # age의 결측값을 0.0으로 대체하고, 모든 값을 정수로 변환
    train_df['age'] = train_df['age'].fillna(0.0).astype(np.int16)
    features.append('age')

    # renta의 결측값을 1.0으로 대체하고, log를 씌워 분포를 변형
    train_df['renta'].fillna(1.0, inplace=True)
    train_df['renta'] = train_df['renta'].map(math.log)
    features.append('renta')

    # 고빈도 100개의 순위를 추출한다.
    # 근데 renta는 가구 수입인데 고빈도가 의미가 있나..?
    train_df['renta_top'] = encode_top(train_df['renta'])
    features.append('renta_top')

    # 결측값 혹은 음수를 0으로 대체하고 나머지 값은 +1.0은 한 이후에 정수로 변환
    # 은행거래 누적 기간
    train_df['antiguedad'] = train_df['antiguedad'].map(
        lambda x: .0 if x < 0 or math.isnan(x) else x + 1.0
        ).astype(np.int16)
    features.append('antiguedad')
    
    # tipodom: 주택 유형, cod_prov: 지방 코드
    # 결측값을 0.0으로 대체하고 정수 변환
    train_df['tipodom'] = train_df['tipodom'].fillna(0.0).astype(np.int8)
    features.append('tipodom')
    train_df['cod_prov'] = train_df['cod_prov'].fillna(0.0).astype(np.int8)
    features.append('cod_prov')

    # 날짜 데이터
    # fecha_dato에서 월/년도를 추출해 정수값 변환
    train_df['fecha_dato_month'] = train_df['fecha_dato'].map(lambda x: int(x.split('-')[1])).astype(np.int8)
    features.append('fecha_dato_month')
    train_df['fecha_dato_year'] = train_df['fecha_dato'].map(lambda x: int(x.split('-')[0])).astype(np.int8)
    features.append('fecha_dato_year')
    # 결측값을 0.0으로 대체하고 fecha_alta에서 월/년도를 추출하여 정수값 변환
    # x.__class__는 결측값을 경우 float을 반환하기 때문에, 결측값 탐지용으로 사용
    # fecha_alta: 고객이 은행과 처음 계약한 날짜
    train_df['fecha_alta_month'] = train_df['fecha_alta'].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])
        ).astype(np.int8)
    features.append('fecha_alta_month')
    train_df['fecha_alta_year'] = train_df['fecha_alta'].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])
        ).astype(np.int8)
    features.append('fecha_alta_year')

    # 날짜 데이터를 월 기준 수치형 변수로 변환
    train_df['fecha_dato_float'] = train_df['fecha_dato'].map(date_to_float)
    train_df['fecha_alta_float'] = train_df['fecha_alta'].map(date_to_float)

    # fecha_dato와 fecha_alto의 월 기준 수치형 변수의 차이값을 파생 변수로 생성
    # 즉 거래 시작 이후 얼마나 지났는지를 저장
    train_df['dato_minus_alta'] = train_df['fecha_dato_float'] - train_df['fecha_alta_float']
    features.append('dato_minus_alta')

    # 날짜 데이터를 월 기준 수치형 변수로 변환
    train_df['int_date'] = train_df['fecha_dato'].map(date_to_int).astype(np.int8)

    # 자체 개발한 one-hot-encoding tngod
    custom_one_hot(train_df, features, 'indresi', {'n': 'N'})
    custom_one_hot(train_df, features, 'indext', {'s': 'S'})
    custom_one_hot(train_df, features, 'conyuemp', {'n': 'N'})
    custom_one_hot(train_df, features, 'sexo', {'h': 'H', 'v': 'V'})
    custom_one_hot(train_df, features, 'ind_empleado', {'a': 'A', 'b': 'B', 'f': 'F', 'n': 'N'})
    custom_one_hot(train_df, features, 'ind_nuevo', {'new': 1})
    custom_one_hot(train_df, features, 'segmento', {
        'top': '01 - TOP', 'particulares': '02 - PARTICULARES', 'universitario': '03 - UNIVERSITARIO'})
    custom_one_hot(train_df, features, 'indfall', {'s': 'S'})
    custom_one_hot(train_df, features, 'indrel', {'1': 1, '99': 99})
    custom_one_hot(train_df, features, 'tiprel_1mes', {'a': 'A', 'i': 'I', 'p': 'P', 'r': 'R'})

    # 결측값을 0.0으로 대체하고, 그 외는 +1.0을 더하고 정수로 변환
    train_df['ind_actividad_cliente'] = train_df['ind_actividad_cliente'].map(
        lambda x: 0.0 if math.isnan(x) else x + 1.0).astype(np.int8)
    features.append('ind_actividad_cliente')

    # 결측값을 0.0으로 대체하고, 'P'를 5로 대체하고 정수로 변환
    train_df['indrel_1mes'] = train_df['indrel_1mes'].map(
        lambda x: 5.0 if x == 'P' else x
    ).astype(float).fillna(0.0).astype(np.int8)
    features.append('indrel_1mes')
    
    # 데이터 전처리/피처 엔지니어링이 1차로 완료된 데이터 프레임과 학습에 사용할 feature 리스트 반환
    return train_df, tuple(features)

# 24개의 금융 변수에 대한 lag 데이터를 직접 생성하는 함수
# int_date를 사용해 24개의 금융변수 값을 step개월만큼 이동시켜 lag 변수 생성
def make_prev_df(train_df, step):
    # 새로운 데이터 프레임에 ncodpers를 추가하고, int_date를 step만큼 이동시킨 값을 넣는다.
    prev_df = pd.DataFrame()
    prev_df['ncodpers'] = train_df['ncodpers']
    prev_df['int_date'] = train_df['int_date'].map(lambda x: x + step).astype(np.int8)

    # 변수명_prev1 형태의 lag 변수 생성
    prod_features = [f'{prod}_prev{step}' for prod in products]
    for prod, prev in zip(products, prod_features):
        prev_df[prev] = train_df[prod]
    
    # prev_df에는 step만큼 이전의 데이터가 들어가 있다.
    # 아직 학습할 df에 머지되진 않음
    return prev_df, tuple(prod_features)

# 기존의 train_df에 lag 데이터를 조인
def join_with_prev(df, prev_df, how):
    # pandas merge 함수를 사용해 join
    df = df.merge(prev_df, on=['ncodpers', 'int_date'], how=how)
    # 24개 금융 변수를 소수형으로 변환
    for f in set(prev_df.columns.values.tolist()) - set(['ncodpers', 'int_date']):
        df[f] = df[f].astype(np.float16)
    return df



def load_data():
    # 데이터 준비에서 통합한 데이터를 읽어온다
    fname = '../input/8th.clean.all.csv'
    train_df = pd.read_csv(fname, dtype=dtypes)

    # products는 util.py에서 정의한 24개 금융 제품 이름
    # 결측값을 0.0으로 대체하고, 정수형으로 변환
    for prod in products:
        train_df[prod] = train_df[prod].fillna(0.0).astype(np.int8)

    # 48개의 변수마다 전처리/피처 엔지니어링 적용
    train_df, features = apply_transforms(train_df)

    prev_dfs = []
    prod_features = None

    use_features = frozenset([1, 2])
    # 1~5까지의 step에 대하여 make_prev_df()를 통해 lag-n 데이터 생성
    for step in range(1, 6):
        prev1_train_df, prod1_features = make_prev_df(train_df, step)
        # 생성한 lag 데이터는 prev_dfs 리스트에 저장
        prev_dfs.append(prev1_train_df)
        # features에는 lag-1,2만 추가
        if step in use_features:
            features + prod1_features
        # prod_features에는 lag-1의 변수명만 저장
        if step == 1:
            prod_features = prod1_features
    return train_df, prev_dfs, features, prod_features

def make_data():
    print('Load data')
    train_df, prev_dfs, features, prod_features = load_data()
    # 여기서는 lag-1만 inner join으로 추가한다. 
    # 이를 거치고 나면 1달 전 이력이 없는 데이터는 삭제된다. -> 아마도 이랬을 때 성능이 좋았던듯
    # 이후에는 left join으로 lag 데이터를 추가해간다.
    for i, prev_df in enumerate(prev_dfs):
        how = 'inner' if i == 0 else 'left'
        train_df = join_with_prev(train_df, prev_df, how=how)

    print('Get std, min, max of lag variable')
    # lag 변수로부터 각 구간별 표준편차, 최댓값, 최솟값을 구하여 데이터에 추가한다.
    # lag 변수의 기초 통계를 명시적으로 변수화하여, 숨겨진 패턴을 찾기 쉽도록 돕는다.
    # 24개 금융 변수에 대해 for loop을 돈다
    for prod in products:
        # [1~3], [1~5], [2~5]의 3개 구간에 대해서 표준편차를 구한다.
        for begin, end in [(1, 3), (1, 5), (2, 5)]:
            prods = [f'{prod}_prev{i}' for i in range(begin, end + 1)]
            mp_df = train_df.as_matrix(columns=prods)
            stdf = f'{prod}_std_{begin}_{end}'

            # np.nanstd로 표준편차를 구하고, features에 신규 파생 변수 이름 추가
            train_df[stdf] = np.nanstd(mp_df, axis=1)
            features += (stdf,)
        # [2~3], [2~5]의 2개 구간에 대해서 최솟값/최댓값을 구한다.
        for begin, end in [(2, 3), (2, 5)]:
            prods = [f'{prod}_prev{i}' for i in range(begin, end + 1)]
            mp_df = train_df.as_matrix(columns=prods)
            minf = f'{prod}_min_{begin}_{end}'
            train_df[minf] = np.nanmin(mp_df, axis=1).astype(np.int8)
            
            maxf = f'{prod}_max_{begin}_{end}'
            train_df[maxf] = np.nanmax(mp_df, axis=1).astype(np.int8)

            features += (minf, maxf,)
    '''
    사용할 변수명의 중복값 여부를 확인하고, 훈련 데이터를 추림
    고객 고유 식별 번호(ncodpers), 정수로 표현한 날짜(int_date), 실제 날짜(fecha_dato) 24개의 금융 변수(products)와 
    학습에 사용하기 위해 전처리/피처 엔지니어링한 변수(features)가 주요 변수
    '''
    leave_columns = ['ncodpers', 'int_date', 'fecha_dato'] + list(products) + list(features)

    # 중복값이 없는지 확인
    assert len(leave_columns) == len(set(leave_columns))

    print('Retrieve main variables')
    # train_df에서 주요 변수만을 추출
    train_df = train_df[leave_columns]
    return train_df, features, prod_features

def make_submission(f, Y_test, C):
    Y_ret = []
    # 파일의 첫 줄에 header를 쓴다
    f.write('ncodpers,added_products\n'.encode('utf-8'))
    # 고객 식별 번호(C)와, 예측 결과물(Y_test)의 for loop
    for c, y_test in zip(C, Y_test):
        # (확률값, 금융 변수명, 금융 변수 id)의 tuple을 구한다
        y_prods = [(y,p,ip) for y,p,ip in zip(y_test, products, range(len(products)))]
        # 확률값을 기준으로 상위 7개 결과만 추출한다
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        # 금융 변수 id를 Y_ret에 저장한다
        Y_ret.append([ip for y,p,ip in y_prods])
        y_prods = [p for y,p,ip in y_prods]
        # 파일에 “고객 식별 번호, 7개의 금융 변수”를 쓴다
        f.write(f'{int(c)},{" ".join(y_prods)}\n'.encode('utf-8'))
    # 상위 7개 예측값을 반환한다
    return Y_ret


def train_predict(all_df, features, prod_features, str_date, cv):
    '''
    all_df: 통합 데이터
    features: 학습에 사용할 변수
    prod_features: 24개 금융 변수
    str_date: 예측 결과물을 산출하는 날짜
    cv: 교체 검증 실행 여부
    '''
    
    # str_date로 예측 결과물을 산출하는 날짜를 지정
    test_date = date_to_int(str_date)
    # 훈련 데이터는 test_date 이전의 모든 데이터를 사용한다.
    train_df = all_df[all_df.int_date < test_date]
    # 테스트 데이터를 통합 데이터에서 분리
    test_df = pd.DataFrame(all_df[all_df.int_date == test_date])

    # 신규 구매 고객만을 훈련 데이터로 추출한다
    X = []
    Y = []
    for i, prod in enumerate(products):
        prev = f'{prod}_prev1'
        # 신규 구매 고객을 prX에 저장
        # 각 prod 별로 신규일 경우에 1 반환 아니면 0 반환
        prX = train_df[(train_df[prod] == 1) & (train_df[prev] == 0)]
        # preY에는 신규 구매에 대한 label 값 저장
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)
    
    XY = pd.concat(X)
    Y = np.hstack(Y)
    # XY는 신규 구매 데이터만 포함된다
    XY['y'] = Y

    # 메모리에서 변수 삭제
    del train_df, all_df

    # 데이터별 가중치를 계산하기 위해서 새로운 변수 (ncodpers + fecha_dato)를 생성
    XY['ncodpers_fecha_dato'] = XY['ncodpers'].astype(str) + XY['fecha_dato']
    uniqs, counts = np.unique(XY['ncodpers_fecha_dato'], return_counts=True)
    # 자연 상수(e)를 통해서, count가 높은 데이터에 낮은 가중치
    weights = np.exp(1 / counts - 1)
    
    # 가중치를 XY 데이터에 추가
    wdf = pd.DataFrame()
    wdf['ncodpers_fecha_dato'] = uniqs
    wdf['counts'] = counts
    wdf['weight'] = weights
    XY = XY.merge(wdf, on='ncodpers_fecha_dato')

    # 교차 검증을 위하여 XY를 훈련:검증(8:2)로 분리
    mask = np.random.rand(len(XY)) < 0.8
    XY_train = XY[mask]
    XY_validate = XY[~mask]

    # 테스트 데이터에서 가중치는 모두 1이다.
    test_df['weight'] = np.ones(len(test_df), dtype=np.int8)

    # 테스트 데이터에서 신규 구매 정답값을 추출
    test_df['y'] = test_df['ncodpers']
    Y_prev = test_df.as_matrix(columns=prod_features)
    for prod in products:
        prev = prod + '_prev1'
        padd = prod + '_add'
        # 신규 구매 여부를 구함
        test_df[padd] = test_df[prod] - test_df[prev]
    
    test_add_mat = test_df.as_matrix(columns=[prod + '_add' for prod in products])
    C = test_df.as_matrix(columns=['ncodpers'])
    test_add_list = [list() for i in range(len(C))]

    # 평가 척도 MAP@7 계산을 위하여, 고객별 신규 구매 정답값을 test_add_list에 기록
    count = 0
    for c in range(len(C)):
        for p in range(len(products)):
            if test_add_mat[c, p] > 0:
                test_add_list[c].append(p)
                count += 1
    
    # 교차 검증에서, 테스트 데이터로 분리된 데이터가 얻을 수 있는 최대 MAP@7 값을 계산한다.
    if cv:
        max_map7 = mapk(test_add_list, test_add_list, 7, 0.0)
        map7coef = float(len(test_add_list)) / float(sum([int(bool(a)) for a in test_add_list]))
        print('Max MAP@7', str_date, max_map7, max_map7 * map7coef)
    
    # LightGBM 모델 학습 후, 예측 결과물을 저장
    Y_test_lgbm = engines.lightgbm(
        XY_train, 
        XY_validate, 
        test_df, 
        features, 
        XY_all=XY, 
        restore=(str_date == '2016-06-28'))
    test_add_list_lightgbm = make_submission(
        io.BytesIO() if cv else gzip.open(f'{str_date}.lightgbm.csv.gz', 'wb'),
        Y_test_lgbm - Y_prev, C
    )
    # 교차 검증일 경우, LightGBM 모델의 테스트 데이터 MAP@7를 출력
    if cv:
        # 정답값인 test_add_list와 lightGBM 모델의 예측값인 test_add_list_lightgbm을
        # mapk 함수에 넣어 평가 척도 점수 확인
        map7lightgbm = mapk(test_add_list, test_add_list_lightgbm, 7, 0.0)
        print(f'LightGBMlib MAP@7 {str_date} {map7lightgbm} {map7lightgbm * map7coef}')

    # XGBoost 모델 학습 후 예측 결과물 저장
    Y_test_xgb = engines.xgboost(
        XY_train, XY_validate, test_df, features, XY_all=XY, restore=(str_date == '2016-06-28'))
    test_add_list_xgboost = make_submission(
        io.BytesIO() if cv else gzip.open(f'{str_date}.xgboost.csv.gz', 'wb'),
        Y_test_lgbm - Y_prev, C
    )
    # 교차 검증일 경우, XGBoost 모델의 테스트 데이터 MAP@7를 출력
    if cv:
        # 정답값인 test_add_list와 XGBoost 모델의 예측값인 test_add_list_xgboost
        # mapk 함수에 넣어 평가 척도 점수 확인
        map7xgboost = mapk(test_add_list, test_add_list_xgboost, 7, 0.0)
        print(f'XBBoost MAP@7 {str_date} {map7xgboost} {map7xgboost * map7coef}')

    # 곱셈 후, 제곱근을 구하는 방식으로 앙상블 수행
    y_test = np.sqrt(np.multiply(Y_test_xgb, Y_test_lgbm))
    # 앙상블 결과물을 저장하고, 테스트 데이터에 대한 MAP@7 출력
    test_add_list_xl = make_submission(
        io.BytesIO() if cv else gzip.open(f'{str_date}.xgboost-lightgbm.csv.gz', 'wb'),
        y_test - Y_prev, C
    )

    # 정답값인 test_add_list와 앙상블 모델의 예측값을 MAPK 함수에 넣어 평가 척도 점수 확인
    if cv:
        map7xl = mapk(test_add_list, test_add_list_xl, 7, 0.0)
        print(f'XGBoost + LightGBM MAP@7 {str_date} {map7xl} {map7xl * map7coef}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')
    parser.add_argument('--load-train-pickle', action='store_true',
                        help='Load preprocessed dataframe from picke file')
    args = parser.parse_args()

    train_df_pickle_path = '../input/8th.feature_engineer.all.pkl'
    cv_meta_pickle_path = '../input/8th.feature_engineer.cv_meta.pkl'
    
    if args.load_train_pickle:
        all_df = pd.read_pickle(train_df_pickle_path)
        with open(cv_meta_pickle_path, 'rb') as f:
            features, prod_features = pickle.load(f)
    else:    
        all_df, features, prod_features = make_data()
        all_df.to_pickle(train_df_pickle_path)
        pickle.dump((features, prod_features), open(cv_meta_pickle_path, 'wb'))
    print('load finish')
    train_predict(all_df, features, prod_features, "2016-05-28", cv=True)
    train_predict(all_df, features, prod_features, "2016-06-28", cv=False)