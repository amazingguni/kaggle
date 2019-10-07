def clean_data(fi, fo, header, suffix):
    # fi: 훈련/테스트 데이터 fd
    # fo: 통합 데이터를 쓰는 경로
    # header: 헤더 쓸지 말지
    # suffix: 테스트 데이터에는 24개의 변수만 있음, 그 곳을 뭘로 채울지

    # csv의 첫줄, 즉 header를 읽어온다
    head = fi.readline().strip('\n').split(',')
    head = [h.strip('"') for h in head]

    # nomprov 변수의 위치를 ip에 저장
    ip = head.index('nomprov')

    if header:
        fo.write(f'{",".join(head)}\n')
    # n은 읽어온 변수의 개수를 의미, (훈련: 48, 테스트: 24)
    n = len(head)
    for line in fi:
        # 파일의 내용을 읽어와서 ,로 분리
        fields = line.strip('\n').split(',')
        
        # nomprov(지역)에 ,를 포함하는 데이터가 존재
        # ,으로 분리된 데이터를 다시 조합
        if len(fields) > n:
            prov = fields[ip] + fields[ip+1]
            del fields[ip]
            fields[ip] = prov
        
        # 데이터 개수가 n개와 동일한지 확인하고 파일에 write, 
        # 테스트 데이터의 경우 suffix는 24개의 공백
        assert len(fields) == n
        fields = [field.strip() for field in fields]
        fo.write(f'{",".join(fields)}{suffix}\n')
    
# 하나의 데이터로 통합하는 코드, 훈련데이터를 쓰고 테스트 데이터를 쓴다.
# 이후부터는 하나의 dataframe만으로 데이터 전처리 진행
with open('../input/8th.clean.all.csv', 'w') as f:
    clean_data(open('../input/train_ver2.csv'), f, True, '')
    comma24 = ',' * 24
    clean_data(open('../input/test_ver2.csv'), f, False, comma24)

    