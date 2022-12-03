import torch
import time


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


#
# def generate_square_subsequent_mask2(sz, device):
#     mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1)
#     mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
#     return mask


def generate_square_subsequent_mask3(sz, device):
    mask = (torch.triu(torch.ones((sz, sz - 1), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_mask = generate_square_subsequent_mask(src_seq_len, device)
    mmr_mask = generate_square_subsequent_mask3(src_seq_len, device)
    return src_mask, tgt_mask, mmr_mask


def get_batch(source, i, bs, bptt):
    seq_len = min(bptt * bs, len(source) - i)
    data = source[i:i + seq_len]
    #    if(seq_len!=bptt*bs):
    #        print("seq_len:",seq_len)
    target = source[i:i + seq_len].reshape(-1)
    return data, target


def batchify(data, bsz, bptt, device):
    # 데이터셋을 bsz 파트들로 나눕니다.
    if (data.size(0) % (bsz * bptt) != 0):
        #        raise ValueError
        #         print(data.size())
        if len(data.shape) > 1:
            data = data.reshape(-1, bptt, data.size(1)).transpose(0, 1).contiguous()
        else:
            data = data.view(-1, bptt).transpose(0, 1).contiguous()
        return data.to(device)
    #    nbatch = data.size(0) // bsz
    # 깔끔하게 나누어 떨어지지 않는 추가적인 부분(나머지들) 은 잘라냅니다.
    #    data = data.narrow(0, 0, nbatch * bsz)
    # 데이터에 대하여 bsz 배치들로 동등하게 나눕니다.
    if len(data.shape) > 1:
        data = data.reshape(bsz, bptt, data.size(1)).transpose(0, 1).contiguous()
    else:
        data = data.view(-1, bptt).transpose(0, 1).contiguous()
    return data.to(device)


def batchify_with_padding(data, bsz, bptt, device):
    # 데이터셋을 bsz 파트들로 나눕니다.

    if data.size(0) % (bsz * bptt) != 0:
        if len(data.shape) > 1:
            data = data.reshape(-1, bptt, data.size(1)).transpose(0, 1).contiguous()
            padtensor = torch.ones((1, data.size(1), data.size(2)))
            padtensor = padtensor.type(torch.LongTensor)
        else:
            data = data.view(-1, bptt).transpose(0, 1).contiguous()
            padtensor = torch.ones((1, data.size(1)))
            padtensor = padtensor.type(torch.LongTensor)
        data = torch.cat([padtensor, data], dim=0)
        # data = torch.LongTensor(data)
        return data.to(device)

    if len(data.shape) > 1:
        data = data.reshape(bsz, bptt, data.size(1)).transpose(0, 1).contiguous()
        padtensor = torch.ones((1, data.size(1), data.size(2)))
        padtensor = padtensor.type(torch.LongTensor)
    else:
        data = data.view(-1, bptt).transpose(0, 1).contiguous()
        padtensor = torch.ones(1, data.size(1))
        padtensor = padtensor.type(torch.LongTensor)
    data = torch.cat([padtensor, data], dim=0)
    return data.to(device)


def train_epoch(model, optimizer, Xtrain_data, Ytrain_data, loss_fn, device, BATCH_SIZE=32, bptt=39):
    model.train()
    losses = 0
    total = 0
    correct = 0
    correct1 = 0
    tot1 = 0
    correct2 = 0
    tot2 = 0
    correct0 = 0
    tot0 = 0
    conf00 = 0
    conf01 = 0
    conf02 = 0
    conf10 = 0
    conf11 = 0
    conf12 = 0
    conf20 = 0
    conf21 = 0
    conf22 = 0

    i = 0
    while i < Xtrain_data.size(0):
        data, org_targets = get_batch(Xtrain_data, i, BATCH_SIZE, bptt)
        if data.isnan().any() or data.isinf().any():
            print(data)
            continue
        targets, _ = get_batch(Ytrain_data, i, BATCH_SIZE, bptt)

        targets = torch.unsqueeze(targets, 1)
        src = batchify(data, BATCH_SIZE, bptt, device)
        tgt = batchify_with_padding(targets, BATCH_SIZE, bptt, device)
        # print(tgt)
        src_input = src[:]
#         print("src shape", src.shape)
#         print("tgt shape", tgt.shape)
        tgt_input = tgt[:-1]
        # print("tgt input shape", tgt_input.shape)
        src_mask, tgt_mask, mmr_mask = create_mask(src_input, tgt_input, device)
        # logits = model(src_input, tgt_input, src_mask, tgt_mask, mmr_mask)

        logits = model(src_input, tgt_input, src_mask, tgt_mask,src_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:]
#         print("src in shape", src_input.shape)
#         print("tgt in shape", tgt_input.shape)
#         print("src shape", src_mask.shape)
#         print("tgt shape", tgt_mask.shape)
#         print("tgt output shape", tgt_out.shape)
#         print("logit shape", logits.shape)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        if (loss.isnan()):
            print('NNNNN')
            print(loss)
        _, predicted = torch.max(logits, -1)
        correct += (tgt_out.squeeze() == predicted).sum().item()
        total += len(predicted) * BATCH_SIZE
        tot0 += (0 == tgt_out.squeeze()).sum().item()
        tot1 += (1 == tgt_out.squeeze()).sum().item()
        tot2 += (2 == tgt_out.squeeze()).sum().item()
        correct0 += ((0 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        correct1 += ((1 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        correct2 += ((2 == predicted) & (2 == tgt_out.squeeze())).sum().item()

        conf00 += ((0 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        conf01 += ((0 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        conf02 += ((0 == predicted) & (2 == tgt_out.squeeze())).sum().item()
        conf10 += ((1 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        conf11 += ((1 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        conf12 += ((1 == predicted) & (2 == tgt_out.squeeze())).sum().item()
        conf20 += ((2 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        conf21 += ((2 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        conf22 += ((2 == predicted) & (2 == tgt_out.squeeze())).sum().item()

        i += targets.size()[0]
    tp0 = conf00
    fp0 = conf01 + conf02
    fn0 = conf10 + conf20
    if (tp0 + fp0 == 0):
        prec0 = 0
    else:
        prec0 = tp0 / (tp0 + fp0)
    if (tp0 + fn0 == 0):
        reca0 = 0
    else:
        reca0 = tp0 / (tp0 + fn0)

    tp1 = conf11
    fp1 = conf10 + conf12
    fn1 = conf01 + conf21

    if (tp1 + fp1 == 0):
        prec1 = 0
    else:
        prec1 = tp1 / (tp1 + fp1)
    if (tp1 + fn1 == 0):
        reca1 = 0
    else:
        reca1 = tp1 / (tp1 + fn1)

    tp2 = conf22
    fp2 = conf20 + conf21
    fn2 = conf02 + conf12

    if (tp2 + fp2 == 0):
        prec2 = 0
    else:
        prec2 = tp2 / (tp2 + fp2)
    if (tp2 + fn2 == 0):
        reca2 = 0
    else:
        reca2 = tp2 / (tp2 + fn2)

    prec = (prec0 + prec1 + prec2) / 3
    reca = (reca0 + reca1 + reca2) / 3
    f1sc = 2 * (prec * reca) / (prec + reca)
    print(total)
    print(correct)
    print("Acc:", correct / total)
    print("Prec", prec)
    print("Recall", reca)
    print("F1", f1sc)
    return losses / Xtrain_data.size(0), [conf00, conf01, conf02, conf10, conf11, conf12, conf20, conf21, conf22]


def evaluate(model, Xtest_data, Ytest_data, loss_fn, device, BATCH_SIZE, bptt):
    model.eval()
    losses = 0
    total = 0
    correct = 0
    correct1 = 0
    tot1 = 0
    correct2 = 0
    tot2 = 0
    correct0 = 0
    tot0 = 0
    conf00 = 0
    conf01 = 0
    conf02 = 0
    conf10 = 0
    conf11 = 0
    conf12 = 0
    conf20 = 0
    conf21 = 0
    conf22 = 0
    stime = time.time()
    with torch.no_grad():
        i = 0
        while (i < Xtest_data.size(0)):
            #         for i in range(0, Xtest_data.size(0) - 1, bptt):

            data, org_targets = get_batch(Xtest_data, i, BATCH_SIZE, bptt)
            if (data.isnan().any() or data.isinf().any()):
                continue
            targets, _ = get_batch(Ytest_data, i, BATCH_SIZE, bptt)
            targets = torch.unsqueeze(targets, 1)
            src = batchify(data, BATCH_SIZE, bptt, device)
            tgt = batchify_with_padding(targets, BATCH_SIZE, bptt, device)

            src_input = src[:]

            tgt_input = tgt[:-1]
            #             print(src_input.size())
            #             print(tgt_input.size())
            src_mask, tgt_mask, _ = create_mask(src_input, tgt_input, device)

            #             logits = model(src_input, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            # logits = model(src_input, tgt_input, src_mask, tgt_mask, mmr_mask)
            logits = model(src_input, tgt_input, src_mask, tgt_mask, src_mask)

            tgt_out = tgt[1:]

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            if (loss.isnan()):
                print(src, tgt_input)
                break
            losses += loss.item()
            _, predicted = torch.max(logits, -1)

            correct += (tgt_out.squeeze() == predicted).sum().item()
            total += len(predicted) * BATCH_SIZE
            tot0 += (0 == tgt_out.squeeze()).sum().item()
            tot1 += (1 == tgt_out.squeeze()).sum().item()
            tot2 += (2 == tgt_out.squeeze()).sum().item()

            conf00 += ((0 == predicted) & (0 == tgt_out.squeeze())).sum().item()
            conf01 += ((0 == predicted) & (1 == tgt_out.squeeze())).sum().item()
            conf02 += ((0 == predicted) & (2 == tgt_out.squeeze())).sum().item()
            conf10 += ((1 == predicted) & (0 == tgt_out.squeeze())).sum().item()
            conf11 += ((1 == predicted) & (1 == tgt_out.squeeze())).sum().item()
            conf12 += ((1 == predicted) & (2 == tgt_out.squeeze())).sum().item()
            conf20 += ((2 == predicted) & (0 == tgt_out.squeeze())).sum().item()
            conf21 += ((2 == predicted) & (1 == tgt_out.squeeze())).sum().item()
            conf22 += ((2 == predicted) & (2 == tgt_out.squeeze())).sum().item()

            i += targets.size()[0]
    etime = time.time()
    print("Time elapsed", etime - stime)
    tp0 = conf00
    fp0 = conf01 + conf02
    fn0 = conf10 + conf20
    if (tp0 + fp0 == 0):
        prec0 = 0
    else:
        prec0 = tp0 / (tp0 + fp0)
    if (tp0 + fn0 == 0):
        reca0 = 0
    else:
        reca0 = tp0 / (tp0 + fn0)

    tp1 = conf11
    fp1 = conf10 + conf12
    fn1 = conf01 + conf21

    if (tp1 + fp1 == 0):
        prec1 = 0
    else:
        prec1 = tp1 / (tp1 + fp1)
    if (tp1 + fn1 == 0):
        reca1 = 0
    else:
        reca1 = tp1 / (tp1 + fn1)

    tp2 = conf22
    fp2 = conf20 + conf21
    fn2 = conf02 + conf12

    if (tp2 + fp2 == 0):
        prec2 = 0
    else:
        prec2 = tp2 / (tp2 + fp2)
    if (tp2 + fn2 == 0):
        reca2 = 0
    else:
        reca2 = tp2 / (tp2 + fn2)

    prec = (prec0 + prec1 + prec2) / 3
    reca = (reca0 + reca1 + reca2) / 3
    f1sc = 2 * (prec * reca) / (prec + reca)
    print(total)
    print(correct)
    print("Acc:", correct / total)
    print("Prec", prec)
    print("Recall", reca)
    print("F1", f1sc)
    return losses / Xtest_data.size(0), correct / total, prec, reca, f1sc, [conf00, conf01, conf02, conf10, conf11,
                                                                            conf12, conf20, conf21, conf22]


from torch.utils.data import DataLoader


def train_epoch_lstm(model, optimizer, Xtrain_data, Ytrain_data, loss_fn, device, BATCH_SIZE, bptt):
    model.train()
    losses = 0
    total = 0
    correct = 0
    correct1 = 0
    tot1 = 0
    correct2 = 0
    tot2 = 0
    correct0 = 0
    tot0 = 0
    conf00 = 0
    conf01 = 0
    conf02 = 0
    conf10 = 0
    conf11 = 0
    conf12 = 0
    conf20 = 0
    conf21 = 0
    conf22 = 0

    #     for batch, i in enumerate(range(0, Xtrain_data.size(0) - 1, BATCH_SIZE*bptt)):
    i = 0
    while (i < Xtrain_data.size(0)):
        data, org_targets = get_batch(Xtrain_data, i, BATCH_SIZE, bptt)
        #        if (data.isnan().any() or data.isinf().any()):
        #            print(data)
        #            continue
        #         _,targets = get_batch(Ytrain_data,i)
        targets, _ = get_batch(Ytrain_data, i, BATCH_SIZE, bptt)
        #         src = src.to(DEVICE)
        #         tgt = tgt.to(DEVICE)
        targets = torch.unsqueeze(targets, 1)
        src = batchify(data, BATCH_SIZE, bptt, device)
        tgt = batchify(targets, BATCH_SIZE, bptt, device)
        #        src=src[:-1]
        #        tgt=tgt[1:]
        logits = model(src)

        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        _, predicted = torch.max(logits, -1)
        correct += (tgt.squeeze() == predicted).sum().item()
        total += len(predicted) * BATCH_SIZE
        tot0 += (0 == tgt.squeeze()).sum().item()
        tot1 += (1 == tgt.squeeze()).sum().item()
        tot2 += (2 == tgt.squeeze()).sum().item()
        correct0 += ((0 == predicted) & (0 == tgt.squeeze())).sum().item()
        correct1 += ((1 == predicted) & (1 == tgt.squeeze())).sum().item()
        correct2 += ((2 == predicted) & (2 == tgt.squeeze())).sum().item()

        conf00 += ((0 == predicted) & (0 == tgt.squeeze())).sum().item()
        conf01 += ((0 == predicted) & (1 == tgt.squeeze())).sum().item()
        conf02 += ((0 == predicted) & (2 == tgt.squeeze())).sum().item()
        conf10 += ((1 == predicted) & (0 == tgt.squeeze())).sum().item()
        conf11 += ((1 == predicted) & (1 == tgt.squeeze())).sum().item()
        conf12 += ((1 == predicted) & (2 == tgt.squeeze())).sum().item()
        conf20 += ((2 == predicted) & (0 == tgt.squeeze())).sum().item()
        conf21 += ((2 == predicted) & (1 == tgt.squeeze())).sum().item()
        conf22 += ((2 == predicted) & (2 == tgt.squeeze())).sum().item()

        i += targets.size()[0]

    tp0 = conf00
    fp0 = conf01 + conf02
    fn0 = conf10 + conf20
    if (tp0 + fp0 == 0):
        prec0 = 0
    else:
        prec0 = tp0 / (tp0 + fp0)
    if (tp0 + fn0 == 0):
        reca0 = 0
    else:
        reca0 = tp0 / (tp0 + fn0)

    tp1 = conf11
    fp1 = conf10 + conf12
    fn1 = conf01 + conf21

    if (tp1 + fp1 == 0):
        prec1 = 0
    else:
        prec1 = tp1 / (tp1 + fp1)
    if (tp1 + fn1 == 0):
        reca1 = 0
    else:
        reca1 = tp1 / (tp1 + fn1)

    tp2 = conf22
    fp2 = conf20 + conf21
    fn2 = conf02 + conf12

    if (tp2 + fp2 == 0):
        prec2 = 0
    else:
        prec2 = tp2 / (tp2 + fp2)
    if (tp2 + fn2 == 0):
        reca2 = 0
    else:
        reca2 = tp2 / (tp2 + fn2)

    prec = (prec0 + prec1 + prec2) / 3
    reca = (reca0 + reca1 + reca2) / 3
    f1sc = 2 * (prec * reca) / (prec + reca)
    print(total)
    print(correct)
    print("Acc:", correct / total)
    print("Prec", prec)
    print("Recall", reca)
    print("F1", f1sc)
    return losses / Xtrain_data.size(0), [conf00, conf01, conf02, conf10, conf11, conf12, conf20, conf21, conf22]


def evaluate_lstm(model, Xtest_data, Ytest_data, loss_fn, device, BATCH_SIZE, bptt):
    model.eval()
    losses = 0
    total = 0
    correct = 0
    correct1 = 0
    tot1 = 0
    correct2 = 0
    tot2 = 0
    correct0 = 0
    tot0 = 0
    conf00 = 0
    conf01 = 0
    conf02 = 0
    conf10 = 0
    conf11 = 0
    conf12 = 0
    conf20 = 0
    conf21 = 0
    conf22 = 0
    with torch.no_grad():

        i = 0
        while (i < Xtest_data.size(0)):
            data, org_targets = get_batch(Xtest_data, i, BATCH_SIZE, bptt)
            if (data.isnan().any() or data.isinf().any()):
                print(data)
                continue
            targets, _ = get_batch(Ytest_data, i, BATCH_SIZE, bptt)
            targets = torch.unsqueeze(targets, 1)
            src = batchify(data, BATCH_SIZE, bptt, device)
            tgt = batchify(targets, BATCH_SIZE, bptt, device)
            #             print(src.shape)
            if (src.shape[1] != BATCH_SIZE):
                break
            #             print(tgt.shape)
            #            src=src[:-1]
            #            tgt=tgt[1:]

            logits = model(src)
            #            print(logits.reshape(-1, logits.shape[-1]).shape,'AA')
            #            print(tgt.reshape(-1).shape,'BB')
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            if (loss.isnan()):
                print(src, tgt_input)
                break
            losses += loss.item()
            _, predicted = torch.max(logits, -1)

            correct += (tgt.squeeze() == predicted).sum().item()
            total += len(predicted) * BATCH_SIZE
            tot0 += (0 == tgt.squeeze()).sum().item()
            tot1 += (1 == tgt.squeeze()).sum().item()
            tot2 += (2 == tgt.squeeze()).sum().item()
            correct0 += ((0 == predicted) & (0 == tgt.squeeze())).sum().item()
            correct1 += ((1 == predicted) & (1 == tgt.squeeze())).sum().item()
            correct2 += ((2 == predicted) & (2 == tgt.squeeze())).sum().item()

            conf00 += ((0 == predicted) & (0 == tgt.squeeze())).sum().item()
            conf01 += ((0 == predicted) & (1 == tgt.squeeze())).sum().item()
            conf02 += ((0 == predicted) & (2 == tgt.squeeze())).sum().item()
            conf10 += ((1 == predicted) & (0 == tgt.squeeze())).sum().item()
            conf11 += ((1 == predicted) & (1 == tgt.squeeze())).sum().item()
            conf12 += ((1 == predicted) & (2 == tgt.squeeze())).sum().item()
            conf20 += ((2 == predicted) & (0 == tgt.squeeze())).sum().item()
            conf21 += ((2 == predicted) & (1 == tgt.squeeze())).sum().item()
            conf22 += ((2 == predicted) & (2 == tgt.squeeze())).sum().item()

            i += targets.size()[0]

    #     print(total,tot0,tot1,tot2)
    #     print(correct)
    tp0 = conf00
    fp0 = conf01 + conf02
    fn0 = conf10 + conf20
    if (tp0 + fp0 == 0):
        prec0 = 0
    else:
        prec0 = tp0 / (tp0 + fp0)
    if (tp0 + fn0 == 0):
        reca0 = 0
    else:
        reca0 = tp0 / (tp0 + fn0)

    tp1 = conf11
    fp1 = conf10 + conf12
    fn1 = conf01 + conf21

    if (tp1 + fp1 == 0):
        prec1 = 0
    else:
        prec1 = tp1 / (tp1 + fp1)
    if (tp1 + fn1 == 0):
        reca1 = 0
    else:
        reca1 = tp1 / (tp1 + fn1)

    tp2 = conf22
    fp2 = conf20 + conf21
    fn2 = conf02 + conf12

    if (tp2 + fp2 == 0):
        prec2 = 0
    else:
        prec2 = tp2 / (tp2 + fp2)
    if (tp2 + fn2 == 0):
        reca2 = 0
    else:
        reca2 = tp2 / (tp2 + fn2)

    prec = (prec0 + prec1 + prec2) / 3
    reca = (reca0 + reca1 + reca2) / 3
    if (prec + reca == 0):
        f1sc = 0
    else:
        f1sc = 2 * (prec * reca) / (prec + reca)
    print(total)
    print(correct)
    print("Acc:", correct / total)
    print("Prec", prec)
    print("Recall", reca)
    print("F1", f1sc)
    return losses / Xtest_data.size(0), correct / total, prec, reca, f1sc, [conf00, conf01, conf02, conf10, conf11,
                                                                            conf12, conf20, conf21, conf22]
