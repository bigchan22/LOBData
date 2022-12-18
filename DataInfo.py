header_df=['ISU_CD', 'ORD_DD', 'ORD_ACPT_NO', 'REGUL_OFFHR_TP_CD', 'BLKTRD_TP_CD', '호가장처리가격',
       'ASKBID_TP_CD', 'MODCANCL_TP_CD', 'ORD_TP_CD', 'ORD_COND_CD', 'INVST_TP_CD', 'ASK_STEP1_BSTORD_PRC',
       'ASK_STEP1_BSTORD_RQTY', 'ASK_STEP2_BSTORD_PRC','ASK_STEP2_BSTORD_RQTY','ASK_STEP3_BSTORD_PRC',
       'ASK_STEP3_BSTORD_RQTY','ASK_STEP4_BSTORD_PRC', 'ASK_STEP4_BSTORD_RQTY','ASK_STEP5_BSTORD_PRC',
       'ASK_STEP5_BSTORD_RQTY',
       'BID_STEP1_BSTORD_PRC', 'BID_STEP1_BSTORD_RQTY', 'BID_STEP2_BSTORD_PRC', 'BID_STEP2_BSTORD_RQTY',
       'BID_STEP3_BSTORD_PRC', 'BID_STEP3_BSTORD_RQTY','BID_STEP4_BSTORD_PRC', 'BID_STEP4_BSTORD_RQTY',
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
pathdir = '/Data/ksqord15081510/'
envpathdir = pathdir + 'EnvData/'
MBR_count_top=[50,  5,  3, 63, 12, 17,  4, 30, 56, 36,  2, 24,  8, 31, 25, 71, 21,
            58, 44,  1, 46, 45, 42, 10, 72, 68, 22, 33,  6, 37]
MBRN_count_top=[( 5,   194),            (36,     1),
            (17,    29),            (12,   100),
            ( 5,   117),            (63,     1),
            (50,    91),            ( 4, 10118),
            ( 4,  9997),            ( 5,   136),
            ( 4, 10121),            (12,   202),
            (63,    15),            ( 2,    83),
            ( 8,   298),            (56, 17990),
            (63,   201),            (50,    64),
            (50,    46),            (50,    80),
            (50,    90),            (58,     1),
            (30, 19920),            (50,    92),
            (44,     1),            ( 5,   499),
            ( 5,   175),            (50,    89),
            (63,    11),            (50,    47)]
MBRNPT_count_top=[(36,     1),            (42,     1),
            (58,     1),            (45,     1),
            (36,     2),            ( 2,   155),
            (17,   700),            (37,     1),
            (33,     1),            (43,     1),
            (44,     1),            ( 2,   999),
            (12,   197),            ( 2,   144),
            (30, 19920),            ( 3,  1800),
            (21, 51090),            (24, 10280),
            (46,   999),            ( 3,  1710),
            (56, 21910),            (54,     1),
            ( 5,   155),            (12,   999),
            (63,    10),            ( 8,     1),
            ( 4, 80045),            (50,     1),
            (10,  1001),            (30, 19910)]
MBRN_sum_top = [( 5,   194),            (12,   100),
            (44,     1),            (17,    29),
            ( 2,    83),            ( 4,  9997),
            (42,     1),            (50,    91),
            (56, 17990),            (12,   202),
            ( 5,   136),            ( 4, 10118),
            (50,    90),            (36,     1),
            (50,    64),            (56, 17988),
            (43,     1),            ( 8,   298),
            (50,    31),            (50,    33),
            ( 4, 10121),            (50,    19),
            (50,    28),            (50,    47),
            (50,    89),            (50,    35),
            (30, 10540),            ( 2,   155),
            ( 3, 91253),            ( 3, 91257)]
MBRNPT_sum_top= [(42,    1),            (36,     1),
            (44,     1),            (43,     1),
            ( 2,   155),            (45,     1),
            (58,     1),            (33,     1),
            (54,     1),            (36,     2),
            (37,     1),            ( 3,  1710),
            (30, 19910),            (41,     1),
            ( 5,   167),            (17,   700),
            (12,   172),            ( 5,   155),
            (30, 19920),            ( 3,  1370),
            ( 2,   144),            ( 6,   570),
            (40,     1),            ( 8,     1),
            (25,  1093),            (35,     1),
            ( 8,   999),            (30, 10620),
            (30, 10640),            (30, 19900)]
MBR_sum_top=[50,  5, 12,  3, 30, 17,  4,  2, 56, 24, 63,  8, 44, 25, 31, 21,  1,
            10, 42, 36, 46, 43, 71, 22, 68, 72, 33,  6, 45, 58]
ISU_list_MBRN_sum_top=['KR7223310004', 'KR7161580006', 'KR7060260007', 'KR7024840001',
       'KR7035720002', 'KR7049180003', 'KR7217270008', 'KR7145020004',
       'KR7083660001', 'KR7031860000', 'KR7030270003', 'KR7192080000',
       'KR7197210008', 'KR7068270008', 'KR7014940001', 'KR7053030003',
       'KR7201490000', 'KR7086900008', 'KR7086390002', 'KR7025440009',
       'KR7064260003', 'KR7036930006', 'KR7067170001', 'KR7033830001',
       'KR7068790005', 'KR7065060006', 'KR7102940004', 'KR7091590000',
       'KR7035480003', 'KR7008800005', 'KR7050110006', 'KR7114570005',
       'KR7036690006', 'KR7240810002', 'KR7178920005', 'KR7130960008',
       'KR7034230003', 'KR7036830008', 'KR7051370005', 'KR7043580000',
       'KR7090360009', 'KR7046890000', 'KR7204840003', 'KR7131370009',
       'KR7056190002', 'KR7001840008', 'KR7224060004', 'KR7033500000',
       'KR7078150000', 'KR7090710005', 'KR7067390005', 'KR7064240005',
       'KR7108790007', 'KR7212560007', 'KR7006920003', 'KR7049120009',
       'KR7022100002', 'KR7109740001', 'KR7100130004', 'KR7215600008',
       'KR7013810007', 'KR7200230001', 'KR7048430003', 'KR7217480003',
       'KR7095610002', 'KR7018000000', 'KR7002800001', 'KR7057880007',
       'KR7131090003', 'KR7141000000', 'KR7048470009', 'KR7222800005',
       'KR7123100000', 'KR7032860009', 'KR7049950009', 'KR7041830001',
       'KR7079650008', 'KR7208140004', 'KR7084110006', 'KR7091700005',
       'KR7041910001', 'KR7265520007', 'KR7024810004', 'KR7031330004',
       'KR7214870008', 'KR7214370009', 'KR7084990001', 'KR7083500009',
       'KR7033100009', 'KR7006910004', 'KR7090460007', 'KR7104040001',
       'KR7066970005', 'KR7005290002', 'KR7064480007', 'KR7053800009',
       'KR7030530000', 'KR7080160005', 'KR7078340007', 'KR7044490001']

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
