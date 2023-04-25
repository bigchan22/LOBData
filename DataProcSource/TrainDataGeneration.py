def preprocess_env_data(env_data, groupmin):
    env_data = append_TM_GP(env_data, groupmin=groupmin)
    env_data = append_ORD_VOL(env_data)
    env_data = append_STEP5(env_data)
    env_data['10단계호가합계잔량'] = env_data['매수10단계호가합계잔량'] + env_data['매도10단계호가합계잔량']
    return env_data


def append_ORD_TM(DF):
    DF['ORD_TM'] = (DF['ORD_ACPT_TM'] // 1e7) * 36e5 + ((DF['ORD_ACPT_TM'] % 1e7) // 1e5) * 6e4 + (
            DF['ORD_ACPT_TM'] % 1e5)
    DF['ORD_TM'] = DF['ORD_TM'] - 9 * 36e5
    return DF


def append_TM_GP(DF, groupmin=10, groupsec=0):
    DF = append_ORD_TM(DF)
    DF['TM_GP'] = DF['ORD_TM'] // (groupmin * 6e4 + groupsec * 1e3)
    return DF


def append_ORD_VOL(DF):
    DF['ORD_VOL'] = DF['ORD_QTY'] * (DF['ORD_PRC'] + DF['직전체결가격']*(DF['ORD_PRC'] == 0))
    return DF
def append_ORD_VOL_ori(DF):
    DF['ORD_VOL'] = DF['ORD_QTY'] * DF['ORD_PRC']
    return DF

def append_STEP5(DF):
    DF['매도5단계호가합계잔량'] = DF['ASK_STEP1_BSTORD_RQTY'] + DF['ASK_STEP2_BSTORD_RQTY'] + DF['ASK_STEP3_BSTORD_RQTY'] + \
                        DF['ASK_STEP4_BSTORD_RQTY'] + DF['ASK_STEP5_BSTORD_RQTY']
    DF['매수5단계호가합계잔량'] = DF['BID_STEP1_BSTORD_RQTY'] + DF['BID_STEP2_BSTORD_RQTY'] + DF['BID_STEP3_BSTORD_RQTY'] + \
                        DF['BID_STEP4_BSTORD_RQTY'] + DF['BID_STEP5_BSTORD_RQTY']
    DF['5단계호가합계잔량'] = DF['매수5단계호가합계잔량'] + DF['매도5단계호가합계잔량']
    return DF


def GetGroupDataFrame(Data, groupcolumns, sumcolumns, meancolumns, lastcolumns):
    aggcolumndict = {}
    for column in sumcolumns:
        aggcolumndict[column] = 'sum'
    for column in meancolumns:
        aggcolumndict[column] = 'mean'
    for column in lastcolumns:
        aggcolumndict[column] = 'last'
    return (Data.groupby(groupcolumns).agg(aggcolumndict)).reset_index()
