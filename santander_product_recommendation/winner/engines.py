import pickle
import xgboost as xgb
import lightgbm as lgbm
from santander_product_recommendation.winner.utils import products


def xgboost(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    # 학습 파라메터
    params = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'min_child_weight': 10,
        'max_depth': 8,
        'silent': 1,
        'nthread': 16,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'num_class': len(products)
    }

    if not restore:
        # 훈련 데이터에서 X, Y, weight를 추출
        X_train = XY_train.as_matrix(columns=features)
        Y_train = XY_train.as_matrix(columns=['y'])
        W_train = XY_train.as_matrix(columns=['weight'])
        # xgboost 전용 데이터 형식으로 변환
        train = xgb.DMatrix(X_train, label=Y_train, feature_names=features, weight=W_train)

        # 검증 데이터도 동일하게 처리
        # 훈련 데이터에서 X, Y, weight를 추출
        X_validate = XY_validate.as_matrix(columns=features)
        Y_validate = XY_validate.as_matrix(columns=['y'])
        W_validate = XY_validate.as_matrix(columns=['weight'])
        # xgboost 전용 데이터 형식으로 변환
        validate = xgb.DMatrix(X_validate, label=Y_validate, feature_names=features, weight=W_validate)

        # XGBoost 모델 학습
        evallist = [(train, 'train'), (validate, 'evel')]
        model = xgb.train(params, train, 1000, evals=evallist, early_stopping_rounds=20)
        # 학습된 모델 저장
        pickle.dump(model, open('next_multi.pickle', 'wb'))
    else:
        # '2016-06-28' 테스트 데이터를 사용할 때에는 사전 학습 모델을 불러옴
        model = pickle.load(open('next_multi.pickle', 'rb'))

    # 교체 검증으로 최적의 트리 개수 정함
    best_ntree_limit = model.best_ntree_limit

    if XY_all is not None:
        # 훈련 데이터에서 X, Y, weight를 추출
        X_all = XY_all.as_matrix(columns=features)
        Y_all = XY_all.as_matrix(columns=['y'])
        W_all = XY_all.as_matrix(columns=['weight'])
        # xgboost 전용 데이터 형식으로 변환
        all_data = xgb.DMatrix(X_all, label=Y_all, feature_names=features, weight=W_all)
        evallist = [(all_data, 'all_data')]
        # 전체 훈련 데이터에는 늘어난 양에 비례해 트리 개수 증가
        best_ntree_limit = int(best_ntree_limit * len(XY_all) / len(XY_train))
        # 모델 학습
        model = xgb.train(params, all_data, best_ntree_limit, evals=evallist)

    # 변수 중요도 출력
    print('Feature importance:')
    for kv in sorted([(k, v) for k, v in zip(features, model.get_fscore().items())], key=lambda kv: kv[1],
                     reverse=True):
        print(kv)

    # 예측에 사용할 테스트 데이터를 XGBoost 전용 데이터로 변환
    # 이 때 weight는 모두 1이기에 별도로 추출하지 않음
    X_test = test_df.as_matrix(columns=features)
    test = xgb.DMatrix(X_test, feature_names=features)
    # 테스트 데이터에 대한 예측 결과물 리턴
    return model.predict(test, ntree_limit=best_ntree_limit)


def lightgbm(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    # 훈련 데이터, 검증 데이터 X, Y, weight 추출 후, LightGBM 전용 데이터로 변환
    train = lgbm.Dataset(
        XY_train[list(features)],
        label=XY_train['y'],
        weight=XY_train['weight'],
        feature_name=features)
    validate = lgbm.Dataset(
        XY_validate[list(features)],
        label=XY_validate['y'],
        weight=XY_validate['weight'],
        feature_name=features,
        reference=train)
    # 학습 파라메터
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 24,
        'metric': {'multi_logloss'},
        'is_training_metric': True,
        'max_bin': 255,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 5,
        'num_threads': 4
    }

    if not restore:
        # XGBoost와 동일하게 훈련/검증 데이터를 기반으로 최적의 트리 개수 계산
        model = lgbm.train(
            params, train, num_boost_round=1000, valid_sets=validate, early_stopping_rounds=20)
        best_iteration = model.best_iteration
        # 학습된 모델과 최적의 트리 개수 정보 저장
        model.save_model('tmp/lgbm.model.txt')
        pickle.dump(best_iteration, open('tmp/lgbm.model.meta', 'wb'))
    else:
        model = lgbm.Boost(model_file='tmp/lgbm.model.txt')
        best_iteration = pickle.load(open('tmp/lgbm.model.meta', 'rb'))

    if XY_all is not None:
        # 전체 훈련 데이터에는 늘어난 양에 비례해 트리 개수 증가
        best_iteration = int(best_iteration * len(XY_all) / len(XY_train))
        # 전체 훈련 데이터에 대한 LightGBM 데이터 생성
        all_train = lgbm.Dataset(
            XY_all[list(features)], label=XY_all['y'], weight=XY_all['weight'], feature_name=features)
        # LightBGM으로 모델 학습
        model = lgbm.train(params, all_train, num_boost_round=best_iteration)
        model.save_model('tmp/lgbm.all.model.txt')

    # LightGBM 모델이 제공하는 변수 중요도 기능을 통해 변수 중요도 출력
    print('fFeature importance by split:')
    for kv in sorted([(k, v) for k, v in zip(features, model.feature_importance('split'))], key=lambda kv: kv[1],
                     reverse=True):
        print(kv)
    print('Feature importance by gain:')
    for kv in sorted([(k, v) for k, v in zip(features, model.feature_importance('gain'))], key=lambda kv: kv[1],
                     reverse=True):
        print(kv)
    # 테스트 데이터에 대한 예측 결과물 리턴
    return model.predict(test_df[list(features)], num_iteration=best_iteration)

