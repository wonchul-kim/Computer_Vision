def parse_model_cfg(cfgfile):
    '''
        * cfgfile: cfg 파일 위치
        * return: dict로 구성된 각 block의 정보를 list로 반환 
    '''
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                # 각 line을 list로
    lines = [x for x in lines if len(x) > 0]      # 빈 line 제거
    lines = [x for x in lines if x[0] != '#']     # 주석 제거
    lines = [x.rstrip().lstrip() for x in lines]   # 좌/우 공백 제거
    
    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':              # block의 시작
            if len(block) != 0:         
                blocks.append(block)    
                block = {}              
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks