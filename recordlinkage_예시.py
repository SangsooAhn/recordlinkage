# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 02:42:23 2021

@author: DL
"""
import pandas as pd
from recordlinkage import Compare
from recordlinkage.index import Block
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os

from utils import (
    file_exists, create_newname_for_matching_in_dataframe, extract_sido, sido_dict, 
    make_index_unique, get_columns_list_in_frame_data, load_frame_data)


@dataclass
class MatchInfo:
    ''' 매칭에 사용되는 정보 정의 '''
    
    # 기본 사항
    idx_col:str     # index
    placename_col:str   # 사업장명
    address_col:str     # 사업장 주소
    business_number_col:str # 사업자등록번호

    # 추가 사항
    sido_col:str = None # 시도 구분
    entity_number_col:str  = None   # 법인등록번호
    longitude_col:str = None    # 경도
    latitude_col:str = None     # 위도
    energy_tj_col:str = None   
    energy_toe_col:str = None  
    address_new_col:str = None   # 비교용 주소 컬럼(네이버 주소 변환 결과)이 존재하는 경우
    entity_name_col:str = None          # 법인명
    entity_name_new_col:str=None        # 비교용 법인명
    placename_new_col:str = None        # 비교용 사업장명

@dataclass
class DisplayInfo:
    ''' 매칭에 직접 사용되지는 않지만 정보 제공이 필요한 정보 정의 '''
    display_cols:list[str] = field(default_factory=list)


@dataclass
class FileInfo:

    # 기본 사항
    data_type:str   # 명세서, 산업DB 등
    filename:str
    match_info:MatchInfo
    encoding:str = None
    display_info:DisplayInfo = None

@dataclass
class MatchingReference:
    '''
    비교 기준이 되는 파일, 한 사업장이 타 자료에서 2개 이상으로 매칭될 수 있으므로 
    신고ID를 기준으로 하지 않고 별도의 KID를 생성하여 관리
    '''

    file_info:FileInfo

    def __post_init__(self):
        self.data = self.data_load()
        self.unmatched = self.preprocess_data()
        self.matched = []

        # for key, value in cols.items():
        #     setattr(self, key, value)

    def data_load(self):
        ''' 파일을 로드, 원활한 비교를 위해서 파일 로드 시 index 설정 '''

        if not file_exists(self.file_info.filename):
            raise FileNotFoundError(f'Check if "{self.file_info.filename}" file exists')

        return pd.read_excel(self.file_info.filename, index_col=self.file_info.match_info.idx_col)
    
    def preprocess_data(self):
        
        '''
        사업장명을 비교에 용이한 형태로 변경
        '''

        data = self.data.copy()
        data = create_newname_for_matching_in_dataframe(
            data, self.file_info.match_info.placename_col, self.file_info.match_info.placename_new_col)
        
        # 시도 컬럼이 존재하지 않을 경우
        if not self.file_info.match_info.sido_col in data.columns:
            data[self.file_info.match_info.sido_col] = \
                data[self.file_info.match_info.address_new_col].map(extract_sido).replace(sido_dict)

        data.to_excel('reference_unmatched_0.xlsx')
        return data
        



@dataclass
class MatchingData:
    '''
    '''
    file_info:FileInfo

    def __post_init__(self):

        self.data = self.data_load()
        self.unmatched = self.preprocess_data()
        self.matched = []
        
    def data_load(self):

        if not file_exists(self.file_info.filename):
            raise FileNotFoundError(f'Check if "{self.file_info.filename}" file exists')

        name = self.file_info.filename
        encoding = self.file_info.encoding
        encoding = 'utf-8' if encoding is None else encoding

        columns = get_columns_list_in_frame_data(name, encoding=encoding)

        total_cols_dict = {}
        total_cols_dict.update(vars(self.file_info.match_info))
        if self.file_info.display_info is not None:
            total_cols_dict.update(vars(self.file_info.display_info))

        # 값이 지정된 컬럼 정보만 추출
        usecols = [value for key, value in total_cols_dict.items() if value in columns]
        return load_frame_data(name, encoding=encoding, usecols=usecols)

    def preprocess_data(self):
        
        '''
        사업장명, 인덱스 정리
        '''

        if self.file_info.match_info.address_new_col is None:
            raise ValueError('비교용 주소컬럼이 지정되지 않았습니다. 파일 정보를 확인해주세요')
          
        # 데이터 복사
        data = self.data.copy()

        # 가독성을 위한 변수 재지정
        sido_col = self.file_info.match_info.sido_col
        address_new_col = self.file_info.match_info.address_new_col
        energy_toe_col = self.file_info.match_info.energy_toe_col
        energy_tj_col = self.file_info.match_info.energy_tj_col
        entity_name_col = self.file_info.match_info.entity_name_col
        entity_name_new_col = self.file_info.match_info.entity_name_new_col
        entity_number_col = self.file_info.match_info.entity_number_col
        placename_col = self.file_info.match_info.placename_col
        placename_new_col = self.file_info.match_info.placename_new_col
        business_number_col = self.file_info.match_info.business_number_col
        longitude_col = self.file_info.match_info.longitude_col
        latitude_col = self.file_info.match_info.latitude_col
        idx_col = self.file_info.match_info.idx_col
        if self.file_info.display_info is not None:
            display_cols = self.file_info.display_info.display_cols
        
        # 에너지사용량 변환 (tj->toe)
        if energy_tj_col is not None:
            data[energy_toe_col] = data[energy_tj_col] * 23.88

        if not address_new_col in data.columns.tolist():
            raise ValueError(f'{address_new_col} 주소_변환 컬럼이 없습니다. ')

        if not longitude_col in data.columns.tolist():
            raise ValueError(f'{longitude_col} 경도 컬럼이 없습니다. ')

        if not latitude_col in data.columns.tolist():
            raise ValueError(f'{latitude_col} 위도 컬럼이 없습니다. ')

        # 시도 추출
        data[sido_col] = data[address_new_col].map(extract_sido).replace(sido_dict)

        # 컬럼 정렬
        cols_to_use = [sido_col, entity_name_col, idx_col, 
            placename_col, entity_number_col, business_number_col,
            address_new_col, longitude_col, latitude_col, energy_toe_col]
        
        if self.file_info.display_info is not None:
            full_cols = cols_to_use if display_cols is None else cols_to_use + display_cols
        else:
            full_cols = cols_to_use 


        cols_to_use = [col for col in full_cols if col in data.columns]
        
        data = data[cols_to_use]
            
        # index_col 중복 여부 확인
        data = make_index_unique(data, idx_col)
        data.set_index(idx_col, inplace=True)
    
        # 사업장 명 정리        
        data = create_newname_for_matching_in_dataframe(data, placename_col, placename_new_col)
        
        # 관리업체명이 지정되어 있다면
        if entity_name_new_col is not None:

            data = create_newname_for_matching_in_dataframe(data, entity_name_col, entity_name_new_col)

            # 사업장명에 관리업체명 추가
            data[placename_new_col] = data.apply(
                lambda row:self.add_entity_name(row[entity_name_new_col], row[placename_new_col]), axis=1)

        data.to_excel('matching_unmatched_0.xlsx')
        return data

    @staticmethod
    def add_entity_name(entity_name:str, placename:str)->str:
        '''
        placename에 entity_name이 포함되어 있지 않다면 
        placename에 관리업체명 추가
        '''
        
        if entity_name in placename:
            pass 
        else:
            placename = ' '.join([entity_name, placename])
        return placename


@dataclass
class MatchingCondition:
    ''' 매칭 조건 '''

    items:Dict[str, float]      # 매칭 항목 및 가중치
    threshold:float     # 매칭 판단 기준


class MatchingConditions:
    ''' 매칭 조건 iterable '''

    def __init__(self, conditions:List[MatchingCondition]=None)->None:
        self.conditions = conditions or []
        self.iteration = 0

    def __iter__(self):
        while self.iteration < len(self.conditions):
            yield self.conditions[self.iteration]
            self.iteration += 1

@dataclass
class IterativeMatching:
    ''' 
    동일한 주소에 2개 이상의 사업장이 존재하는 등 사람이 매칭판정을 해야 하는 경우 존재
    위와 같은 경우, 매칭판정이 완료된 사업장은 더 이상 다른 사업장과 매칭이 필요하지 않음
    만약 매칭이 완료된 사업장을 제외한 나머지 사업장에 대해 매칭을 반복한다면
    매칭 시간을 단축하고, 매칭판정 경우의 수를 줄일 수 있음

    '''
    reference:MatchingReference # KID를 가지는 기준 데이터
    matching_data:MatchingData  # 매칭 대상 데이터
    matching_conditions:MatchingConditions    # 매칭 조건
    block:str       # recordlinkage block    
    iteration:int = 0   # 반복횟수
    matched:pd.DataFrame = None # 매칭조건에 해당하는 자료

    def __str__(self):
        iteration_str = f'비교 횟수 {self.iteration}회 \n'
        reference_str = f'reference {self.reference.file_info.filename} \n'
        target_str = f'matching data {self.matching_data.file_info.filename} \n'

        return ''.join([iteration_str, reference_str, target_str])

    def matching(self, result_name = '매칭결과'):
        ''' 매칭 조건에 따라 매칭 수행 후 결과를 self.matched에 저장 '''

        for matching_condition in self.matching_conditions:

            reference_unmatched = self.reference.unmatched.copy()
            matching_unmatched = self.matching_data.unmatched.copy()

            # 비교기 생성
            compare = Compare()
            block_indexer = Block(self.block)
            pairs = block_indexer.index(reference_unmatched, matching_unmatched)

            # 결과 정리용 컬럼
            ref_cols = []
            comp_cols = []
            labels = []     # 항목별 점수

            for item in matching_condition.items.keys():
                
                if item in ['사업장명', '사업자등록번호', '주소']:

                    if item == '사업장명':

                        ref_col = self.reference.file_info.match_info.placename_new_col

                        comp_col = self.matching_data.file_info.match_info.placename_new_col

                    elif item == '사업자등록번호':

                        ref_col = self.reference.file_info.match_info.business_number_col

                        comp_col = self.matching_data.file_info.match_info.business_number_col

                    elif item == '주소':

                        ref_col = self.reference.file_info.match_info.address_new_col

                        comp_col = self.matching_data.file_info.match_info.address_new_col

                    # 오류 방지를 위한 문자열 변환
                    reference_unmatched.loc[:, ref_col] = reference_unmatched.loc[:, ref_col].astype(str)

                    matching_unmatched.loc[:, comp_col] = matching_unmatched.loc[:, comp_col].astype(str)

                    compare.string(ref_col, comp_col, method='jarowinkler', label=item)
                    ref_cols.append(ref_col)
                    comp_cols.append(comp_col)
                    labels.append(item)

                elif item == '에너지사용량':
                    ref_col = self.reference.file_info.match_info.energy_toe_col

                    comp_col = self.matching_data.file_info.match_info.energy_toe_col

                    # 오류 방지를 위한 수치형 변환
                    reference_unmatched.loc[:, ref_col] = reference_unmatched.loc[:, ref_col].astype(float)

                    matching_unmatched.loc[:, comp_col] = matching_unmatched.loc[:, comp_col].astype(float)

                    compare.numeric(ref_col, comp_col, method='gauss', offset=100, scale=100, label=item)
                    ref_cols.append(ref_col)
                    comp_cols.append(comp_col)
                    labels.append(item)

                elif item == 'GPS':

                    ref_lat_col = self.reference.file_info.match_info.latitude_col
                    ref_lng_col = self.reference.file_info.match_info.longitude_col

                    comp_lat_col = self.matching_data.file_info.match_info.latitude_col
                    comp_lng_col = self.matching_data.file_info.match_info.longitude_col

                    compare.geo(ref_lat_col, ref_lng_col, comp_lat_col, comp_lng_col, method='exp', scale=0.1, offset=0.01, label=item)

                    ref_cols.append(ref_lat_col)
                    ref_cols.append(ref_lng_col)

                    comp_cols.append(comp_lat_col)
                    comp_cols.append(comp_lng_col)
                    labels.append(item)

            matching_vectors = compare.compute(
                pairs, reference_unmatched, matching_unmatched)
            
            weights = [value for value in matching_condition.items.values()]
            
            try:
                scores = np.average(
                    matching_vectors.values,
                    axis=1,
                    weights=weights)

                matching_vectors['score'] = scores

            except ValueError:
                print("비교 기준명 확인 : '사업장명', '사업자등록번호', '주소', '에너지사용량', 'GPS' 등")

            # 매칭 검토 결과 저장

            ref_index = pairs.get_level_values(0)
            ref_frame = reference_unmatched.loc[ref_index, ref_cols]
            ref_frame.columns = ['기준_' + col for col in ref_frame.columns]

            comp_index = pairs.get_level_values(1)
            comp_frame = matching_unmatched.loc[comp_index, comp_cols]
            comp_frame.columns = ['비교_' + col for col in comp_frame.columns]

            total_frame = pd.concat([
                ref_frame.reset_index(), 
                comp_frame.reset_index(), 
                matching_vectors.reset_index()], axis=1)
            # total_frame.to_excel('total_frame.xlsx')

            total_cols = ['score']
            for ref_col, comp_col, label in zip(ref_frame.columns, comp_frame.columns, labels):
                
                # GPS는 위도, 경도가 별도의 컬럼
                if label != 'GPS':

                    total_cols.append(ref_col)
                    total_cols.append(comp_col)
                    total_cols.append(label)                    

                else:

                    total_cols.append(ref_col)
                    total_cols.append(comp_col)
                    total_cols.append(ref_col)
                    total_cols.append(comp_col)
                    total_cols.append(label)

            total_frame = total_frame[total_cols]

            total_frame = pd.DataFrame(
                data = total_frame.values,
                index = pairs,
                columns = total_cols
            )
            # total_frame.to_excel('total_frame1.xlsx')


            # 기준을 만족하는 매칭 결과가 존재할 경우
            if (total_frame.score > matching_condition.threshold).sum()>0:
                
                # total_frame['판정'] = ''
                is_matched = total_frame.score > matching_condition.threshold
                print(f'{is_matched.sum()} records matched out of possible {len(total_frame)} cases')
                reference_matched_index = total_frame.loc[is_matched].index.get_level_values(0)
                matching_data_matched_index = total_frame.loc[is_matched].index.get_level_values(1)

                self.matched = (
                    total_frame.loc[is_matched] 
                    if self.matched is None 
                    else pd.concat([self.matched, total_frame.loc[is_matched]], axis=0))

                is_matched = self.reference.unmatched.index.isin(reference_matched_index)
                self.reference.unmatched = self.reference.unmatched.loc[~is_matched]

                is_matched = self.matching_data.unmatched.index.isin(matching_data_matched_index)
                self.matching_data.unmatched = self.matching_data.unmatched.loc[~is_matched]

                
                result_filename = f'{result_name}_{self.iteration}.xlsx'
                total_frame[total_frame.score > matching_condition.threshold].to_excel(result_filename)
            else:
                print('None matched')
                total_frame = None


if __name__ == '__main__':

    os.chdir(r'D:\python_dev\recordlinkage\data\220225 닐슨 원데이터')
    # ref_file_info = FileInfo(
    #     data_type = '신고',
    #     filename = 'master.xlsx',
    #     idx_col = 'KID',
    #     placename_col = '사업장명',
    #     placename_new_col = '사업장명_변경',
    #     address_col = '사업장 주소',
    #     address_new_col = '주소_변환',
    #     business_number_col = '사업자등록번호',

    #     sido_col = '시도',
    #     entity_number_col = '법인등록번호',
    #     longitude_col = '경도',
    #     latitude_col = '위도',
    #     energy_toe_col = '에너지(toe)')

    # reference = MatchingReference(ref_file_info)

    # myeng_se_file_info = FileInfo(
    #     data_type = '명세서',
    #     filename = '02.사업장 일반정보_전체업체_2020100100_.csv',
    #     idx_col = '사업장 일련번호',
    #     encoding = 'cp949',

    #     sido_col = '시도',
    #     placename_col = '사업장 명',
    #     placename_new_col = '사업장명_변경',
    #     entity_name_col = '관리업체명',
    #     entity_name_new_col = '관리업체명_변경',

    #     address_col = '사업장 소재지',
    #     address_new_col = '주소_변환',

    #     business_number_col = '사업자 등록번호',
    #     entity_number_col = '법인 등록번호',

    #     longitude_col = '경도',
    #     latitude_col = '위도',
    #     energy_tj_col = '합계(TJ)', 
    #     energy_toe_col = '합계(toe)',
    #     )

    # matching_data = MatchingData(myeng_se_file_info)

    # matching_condition = MatchingCondition(
    
    #     # 항목, 가중치
    #     items = {
    #         '사업장명':15,
    #         '사업자등록번호':15,
    #         '주소':50,
    #         '에너지사용량':10,
    #         'GPS':20,
    #     },
    #     # 0~1 사이의 값 설정
    #     threshold = 0.7,

    # )
    
    # iterative_matching = IterativeMatching(
    #     reference, 
    #     matching_data,
    #     matching_condition,
    #     '시도'
    # )

    # iterative_matching.matching()

    ref_match_info = MatchInfo(
        idx_col = '업체코드',
        placename_col = '기관명',
        placename_new_col = '사업장명_변경',
        address_col = '주소',
        address_new_col = '주소',
        business_number_col = '사업자등록번호',
        sido_col = '시도',
        entity_number_col = None,
        longitude_col = '경도',
        latitude_col = '위도',
        energy_toe_col = None
    )

    ref_file_info = FileInfo(
        data_type = '신고',
        filename = '신고20_주소변환결과_사용량.xlsx',
        match_info = ref_match_info)

    reference = MatchingReference(ref_file_info)

    matching_match_info = MatchInfo(

        idx_col = '조사표번호',
        sido_col = '시도',
        placename_col = '사업장명',
        placename_new_col = '사업장명_변경',
        entity_name_col = None,
        entity_name_new_col = None,

        address_col = '주소',
        address_new_col = '주소',

        business_number_col = '사업자변호',
        entity_number_col = None,

        longitude_col = '경도',
        latitude_col = '위도',
        energy_tj_col = None, 
        energy_toe_col = None
    )
    # matching_display_info = DisplayInfo(
    #     display_cols=['전력(MWh)_수송제외', '열량(toe,전력최종기준)_수송제외', '배출량(tCO2)_수송제외', '열량(toe,전력1차기준)_수송제외']
        
    # )

    matching_file_info = FileInfo(
        data_type = '산업',
        filename = '산업20_주소변환결과_사용량.xlsx',
        encoding = 'cp949',
        match_info = matching_match_info,
        # display_info = myeng_se_display_info
    )

    matching_data = MatchingData(matching_file_info)

    matching_conditions = MatchingConditions(
    
        [
            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업장명':30,
                    '주소':70
                },
                # 0~1 사이의 값 설정
                threshold = 0.98),

            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업자등록번호':30,
                    '주소':70
                },
                # 0~1 사이의 값 설정
                threshold = 0.98),

            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업장명':30,
                    '주소':70
                },
                # 0~1 사이의 값 설정
                threshold = 0.90),

            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업자등록번호':30,
                    '주소':70
                },
                # 0~1 사이의 값 설정
                threshold = 0.90),

            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업장명':20,
                    '사업자등록번호':30,
                    '주소':50
                },
                # 0~1 사이의 값 설정
                threshold = 0.90),

            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업장명':20,
                    '사업자등록번호':30,
                    '주소':50
                },
                # 0~1 사이의 값 설정
                threshold = 0.80),

            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업장명':20,
                    '사업자등록번호':30,
                    '주소':50
                },
                # 0~1 사이의 값 설정
                threshold = 0.60),

            MatchingCondition(
                # 항목, 가중치
                items = {
                    '사업장명':20,
                    '사업자등록번호':30,
                    '주소':50
                },
                # 0~1 사이의 값 설정
                threshold = 0.50),


            # MatchingCondition(
            #     # 항목, 가중치
            #     items = {
            #         '사업자등록번호':50,
            #         '주소':50,
            #     },
            #     # 0~1 사이의 값 설정
            #     threshold = 0.9)
        ]

    )


    iterative_matching = IterativeMatching(
        reference, 
        matching_data,
        matching_conditions,
        block = '시도'
    )

    iterative_matching.matching()
    iterative_matching.matched.to_excel('matched_신고_산업.xlsx')