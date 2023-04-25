
groupmin = 10
PathDir = '/Data/ksqord_0307/'
EnvPathDir = PathDir + 'EnvData/'
TrainPathDir = '/Data/LOBData/TrainData/'
datasubfix = "0307Trainset2"



envdataoption = "last" #"last" or "mean"
envdatasubfix = str(groupmin) + "min_"+ envdataoption
CancelCount = True
ISUlist= ['KR7044180008', 'KR7024810004', 'KR7096040001', 'KR7036630002',
                'KR7043100007', 'KR7049470008', 'USU652221081', 'KR7023430002',
                'KR7017680000', 'KR7039790001']

MBRNlist=[(50,None),
(5,None),
(2,None),
(12,None),
(3,None),
(36,None),
(17,None),
(4,None),
(63,None),
(56,None)]




header_df = ['ISU_CD', 'ORD_DD', 'ORD_ACPT_NO', 'REGUL_OFFHR_TP_CD', 'BLKTRD_TP_CD', '호가장처리가격',
             'ASKBID_TP_CD', 'MODCANCL_TP_CD', 'ORD_TP_CD', 'ORD_COND_CD', 'INVST_TP_CD', 'ASK_STEP1_BSTORD_PRC',
             'ASK_STEP1_BSTORD_RQTY', 'ASK_STEP2_BSTORD_PRC', 'ASK_STEP2_BSTORD_RQTY', 'ASK_STEP3_BSTORD_PRC',
             'ASK_STEP3_BSTORD_RQTY', 'ASK_STEP4_BSTORD_PRC', 'ASK_STEP4_BSTORD_RQTY', 'ASK_STEP5_BSTORD_PRC',
             'ASK_STEP5_BSTORD_RQTY',
             'BID_STEP1_BSTORD_PRC', 'BID_STEP1_BSTORD_RQTY', 'BID_STEP2_BSTORD_PRC', 'BID_STEP2_BSTORD_RQTY',
             'BID_STEP3_BSTORD_PRC', 'BID_STEP3_BSTORD_RQTY', 'BID_STEP4_BSTORD_PRC', 'BID_STEP4_BSTORD_RQTY',
             'BID_STEP5_BSTORD_PRC', 'BID_STEP5_BSTORD_RQTY', 'ORD_ACPT_TM', 'ORD_QTY', 'ORD_PRC',
             '호가우선순위번호', 'MBR_NO', 'BRN_NO', 'CNTR_CD', 'TRST_PRINC_TP_CD', 'FORNINVST_TP_CD',
             'ORD_MEDIA_TP_CD', '회원사주문시각', '예상체결가격', '예상체결수량', '매도총호가잔량', '매수총호가잔량',
             '호가체결접수순서번호', '직전체결가격', '누적체결수량', '누적거래대금', 'AGG_TM', '시가', '고가', '저가',
             'ORGN_ORD_ACPT_NO', '시장구분코드', '자동취소처리구분코드', 'MKTSTAT_TP_CD', '매도10단계호가합계잔량',
             '매수10단계호가합계잔량', 'PT_TP_CD']
env_columns = ['REGUL_OFFHR_TP_CD',
               'ASK_STEP1_BSTORD_PRC',
               'ASK_STEP1_BSTORD_RQTY', 'ASK_STEP2_BSTORD_PRC', 'ASK_STEP2_BSTORD_RQTY', 'ASK_STEP3_BSTORD_PRC',
               'ASK_STEP3_BSTORD_RQTY', 'ASK_STEP4_BSTORD_PRC', 'ASK_STEP4_BSTORD_RQTY', 'ASK_STEP5_BSTORD_PRC',
               'ASK_STEP5_BSTORD_RQTY',
               'BID_STEP1_BSTORD_PRC', 'BID_STEP1_BSTORD_RQTY', 'BID_STEP2_BSTORD_PRC', 'BID_STEP2_BSTORD_RQTY',
               'BID_STEP3_BSTORD_PRC', 'BID_STEP3_BSTORD_RQTY', 'BID_STEP4_BSTORD_PRC', 'BID_STEP4_BSTORD_RQTY',
               'BID_STEP5_BSTORD_PRC', 'BID_STEP5_BSTORD_RQTY',
               '회원사주문시각', '매도총호가잔량', '매수총호가잔량',
               '직전체결가격', '시가', '고가', '저가',
               '시장구분코드', 'MKTSTAT_TP_CD', '매도10단계호가합계잔량',
               '매수10단계호가합계잔량']

###########################################train data gen


ord_columns = ['ISU_CD', 'ORD_DD', 'ORD_ACPT_NO', 'BLKTRD_TP_CD', '호가장처리가격', 'ASKBID_TP_CD', 'MODCANCL_TP_CD',
               'ORD_TP_CD', 'ORD_COND_CD', 'INVST_TP_CD', 'ORD_ACPT_TM', 'ORD_QTY', 'ORD_PRC', '호가우선순위번호', 'MBR_NO',
               'BRN_NO',
               'CNTR_CD', 'TRST_PRINC_TP_CD', 'FORNINVST_TP_CD', 'ORD_MEDIA_TP_CD', '예상체결가격', '예상체결수량', '호가체결접수순서번호',
               '누적체결수량', '누적거래대금', 'AGG_TM', 'ORGN_ORD_ACPT_NO', '자동취소처리구분코드', 'PT_TP_CD']

feat_cols = ['매도5단계호가합계잔량', '매수5단계호가합계잔량', '매도10단계호가합계잔량',
             '매수10단계호가합계잔량', '매도총호가잔량', '매수총호가잔량', '고가', '저가',
             '시가', '직전체결가격', 'NET_ORD_QTY2']

groupcolumns = ['ORD_DD', 'ISU_CD', 'TM_GP']
sumcolumns = ['NET_ORD_QTY', 'NET_ORD_VOL']
meancolumns = ['ORD_PRC']
lastcolumns = []


divdict = {}
loglist = []
loglist += ['고가', '저가', '직전체결가격']
minmaxnormlist = []
minmaxnormlist += ['매도총호가잔량', '매수총호가잔량', '고가', '저가',
                   '직전체결가격']


'''
5 194 5    대우증권
Name: MKTPARTC_KOR_ABBRV, dtype: object
2 155 53    신한투자
Name: MKTPARTC_KOR_ABBRV, dtype: object
12 100 7    NH투자증권(우리)
Name: MKTPARTC_KOR_ABBRV, dtype: object
17 29 36    현대증권
Name: MKTPARTC_KOR_ABBRV, dtype: object
42 1 172    CS증권
Name: MKTPARTC_KOR_ABBRV, dtype: object
44 1 282    메릴린치
Name: MKTPARTC_KOR_ABBRV, dtype: object
50 92 4    키움증권
Name: MKTPARTC_KOR_ABBRV, dtype: object
2 83 53    신한투자
Name: MKTPARTC_KOR_ABBRV, dtype: object
4 10118 67    대신증권
Name: MKTPARTC_KOR_ABBRV, dtype: object
8 298 273    유진증권
Name: MKTPARTC_KOR_ABBRV, dtype: object
5 194 Series([], Name: MKTPARTC_KOR_ABBRV, dtype: object) Series([], Name: BRN_NM, dtype: object)
2 155 364    신한투자
Name: MKTPARTC_KOR_ABBRV, dtype: object 364    영업10부
Name: BRN_NM, dtype: object
12 100 8    NH투자증권(우리)
Name: MKTPARTC_KOR_ABBRV, dtype: object 8    스마트
Name: BRN_NM, dtype: object
17 29 4939    현대증권
Name: MKTPARTC_KOR_ABBRV, dtype: object 4939    YF사이버
Name: BRN_NM, dtype: object
42 1 172    CS증권
Name: MKTPARTC_KOR_ABBRV, dtype: object 172    고객
Name: BRN_NM, dtype: object
44 1 282    메릴린치
Name: MKTPARTC_KOR_ABBRV, dtype: object 282    서울
Name: BRN_NM, dtype: object
50 92 Series([], Name: MKTPARTC_KOR_ABBRV, dtype: object) Series([], Name: BRN_NM, dtype: object)
2 83 860    신한투자
Name: MKTPARTC_KOR_ABBRV, dtype: object 860    신한은행
Name: BRN_NM, dtype: object
4 10118 44434    대신증권
Name: MKTPARTC_KOR_ABBRV, dtype: object 44434    다이렉트지점
Name: BRN_NM, dtype: object
8 298 Series([], Name: MKTPARTC_KOR_ABBRV, dtype: object) Series([], Name: BRN_NM, dtype: object)
'''
