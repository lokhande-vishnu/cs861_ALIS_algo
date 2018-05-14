import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Using device', device)

trainsize = 50000
testsize = 50000
label_fraction = 0.001
h_star = torch.tensor([0.5, -0.5]).to(device)
m = 10 # Number of samples to draw in a given iteration


'''
Put a bound on the loss 
weighted loss in the algo

trainsize = 5000
testsize = 10000
label_fraction = 0.001
h_star = [0.5, -0.5]
m = 5 # Number of samples to draw in a given iteration

'''

def generateData():
    xtrain = torch.empty(trainsize, 2).uniform_(-1,1).to(device)
    train = torch.empty(trainsize, 3).to(device)
    for i in range(len(xtrain)):
        if torch.dot(xtrain[i], h_star) >= 0:
            train[i][0], train[i][1], train[i][2] = xtrain[i][0], xtrain[i][1], 1.0
        else:
            train[i][0], train[i][1], train[i][2] = xtrain[i][0], xtrain[i][1], -1.0

    xtest = torch.empty(testsize, 2).uniform_(-1,1).to(device)
    test = torch.empty(testsize, 3).to(device)
    for i in range(len(xtest)):
        if torch.dot(xtest[i], h_star) >= 0:
            test[i][0], test[i][1], test[i][2] = xtest[i][0], xtest[i][1], 1.0
        else:
            test[i][0], test[i][1], test[i][2] = xtest[i][0], xtest[i][1], -1.0

    return train, test

def breakData(train):
    S_len = int(label_fraction*len(train))
    U_len = len(train) - S_len
    S, U = torch.split(train, [S_len, U_len])
    return S, U

def perceptron_train(ltrain, w_init, ltrain_pt, eta_):
    w = w_init
    eta = eta_
    epochs = 1

    #print('training sum of pt', sum(ltrain_pt))
    for t in range(epochs):
        for i in range(len(ltrain)):
            x = ltrain[i][:2]
            label = ltrain[i][2]
            if (torch.dot(x, w)*label) <= 0:
                w += 1.0*eta*x*label/ltrain_pt[i]
                #print('CORRECTION', 1.0*eta*x*label/ltrain_pt[i])
    return w

def perceptron_test(data, w):
    diff = torch.mv(data[:, :2], w) * data[:, 2]
    error = torch.sum(diff[diff <= 0])
    error_val = torch.sum(diff <= 0)
    #print(diff)
    #print(torch.clamp(diff, max=0.0))
    #print(torch.sum(torch.clamp(diff, max=0.0)))
    return -error.item(), error_val.item()

def compute_y_pseudo(U, w_pred):
    #y_pseudo = - torch.sign(torch.mv(U[:,:2], w_pred))
    temp = torch.mv(U[:,:2], w_pred)
    temp[temp > 0] = + 1.0
    temp[temp <= 0] = - 1.0
    y_pseudo = - temp
    
    return y_pseudo

def pseudo_loss(U, pt, y_pseudo, w_pred):
    l = torch.clamp(1 - y_pseudo *  torch.mv(U[:,:2], w_pred), min = 0)
    '''
    print('l', l)
    print('w_pred', w_pred)
    print('mm', torch.mv(U[:,:2], w_pred))
    '''
    
    l_sq = l**2
    l_sum = torch.sum(torch.div(l_sq, pt))
    return l_sum.item()
    '''
    t = 0
    for i in range(len(U)):
        
        t += 1.0*np.square(torch.max(1-y_pseudo[i]*torch.dot(w_pred, U[i,:2]), 0)) / pt[i]
    return t
    '''
        

def compute_pt(U, w_pred, y_pseudo):

    l = torch.clamp(1 - y_pseudo *  torch.mv(U[:,:2], w_pred), min = 0)
    pt = torch.div(l, torch.sum(l))
    
    #print('sumn of pt\n', torch.sum(pt))
    #print('\n')
    return pt

def sample(U, pt):
    index_V = list(set(np.random.choice(np.arange(len(U)), m, pt.tolist())))
    V = U[index_V]
    V_pt = pt[index_V]
    return V, index_V, V_pt

def update_sets(U, S, index_V):
    
    S = torch.cat((S, U[index_V]), 0)

    inv_index_V = list(set(np.arange(len(U)))-set(index_V))
    U = U[inv_index_V]

    return U, S


def runALIS(train, test, n_sample_draws):
    # ALIS algorithm
    S, U = breakData(train)
   
    #print('\n Running  ALIS algorithm ...')
    #print('Size of labelled dataset:%d, Size of unlabelled dataset:%d' % (len(S), len(U)))

    eta_ = 1e-5 #1e-5 #
    pt_init = [(1.0/len(S))]*len(S)
    w_init = torch.zeros(2).to(device)
    w_pred = perceptron_train(S, w_init, pt_init, eta_)
    #print('starting test error', perceptron_test(S, w_pred).item())

    test_errors_alis = []
    test_errors_val_alis = []
    true_loss_alis = []
    pseudo_loss_alis = []
    for k in range(n_sample_draws):
        #print('Jere', n_sample_draws)
        y_pseudo = compute_y_pseudo(U, w_pred)
        pt = compute_pt(U, w_pred, y_pseudo)

        #print('before: lenU:%d, lenS:%d, lenyp:%d, lenpt:%d' % (len(U), len(S), len(y_pseudo), len(pt)))
        V, index_V, V_pt =  sample(U, pt)
        #print('w_pred_before', w_pred)
        w_pred = perceptron_train(V, w_pred, V_pt, eta_)

        '''
        print('\n')
        print('### DEBUGGING %d' % k)
        print('Y_PSEUDO', y_pseudo)
        print('VAL_YPS', np.log(pseudo_loss(U, pt, y_pseudo, w_pred)))
        print('\n')
        print('Y_TRUE', U[:,2])
        print('VAL_YTR', np.log(pseudo_loss(U, pt, U[:,2], w_pred)))
        print('END DEBUGGING ###')
        print('\n')
        '''
        
        pseudo_loss_alis.append(np.log(pseudo_loss(U, pt, y_pseudo, w_pred)))
        true_loss_alis.append(np.log(pseudo_loss(U, pt, U[:,2], w_pred)))
        U, S = update_sets(U, S, index_V)
        #print('w_pred_after', w_pred)
        #print('after: lenU:%d, lenS:%d, lenyp:%d, lenpt:%d' % (len(U), len(S), len(y_pseudo), len(pt)))
        
        error_k, error_val = perceptron_test(test, w_pred)
        #print('ALIS - iteration: %d, error: %f' % (k, error_k))
        test_errors_alis.append(error_k)
        test_errors_val_alis.append(error_val)

    #print(w_pred[0].item(), w_pred[1].item())
    #return test_errors_val_alis
    return pseudo_loss_alis, true_loss_alis, test_errors_val_alis


def runRS(train, test, n_sample_draws):
    
    # Random Sampling algorithm
    S, U = breakData(train)
    #print('Running  random sampling algorithm ...')
    #print('Size of labelled dataset:%d, Size of unlabelled dataset:%d' % (len(S), len(U)))

    #eta_ = 0.00000000001
    eta_ = 1e-1
    pt_init = [1.0/len(S)]*len(S)
    w_init = torch.zeros(2).to(device)
    w_pred = perceptron_train(S, w_init, pt_init, eta_)
    #print('starting test error', perceptron_test(S, w_pred).item())

    test_errors_rs = []
    pseudo_loss_rs = []
    test_errors_val_rs = []
    true_loss_rs = []
    for k in range(n_sample_draws):
        y_pseudo = compute_y_pseudo(U, w_pred)
        pt = torch.tensor([1.0/len(U)]*len(U)).to(device)

        #print('before: lenU:%d, lenS:%d, lenyp:%d, lenpt:%d' % (len(U), len(S), len(y_pseudo), len(pt)))
        V, index_V, _ =  sample(U, pt)
        V_pt = [1]*len(V)
        #print('w_pred_before', w_pred)
        w_pred = perceptron_train(V, w_pred, V_pt, eta_)
        pseudo_loss_rs.append(np.log(pseudo_loss(U, pt, y_pseudo, w_pred)))
        true_loss_rs.append(np.log(pseudo_loss(U, pt, U[:, 2], w_pred)))
        #print('HERE........', pseudo_loss(U, pt, y_pseudo, w_pred))
        U, S = update_sets(U, S, index_V)
        #print('w_pred_after', w_pred)
        #print('after: lenU:%d, lenS:%d, lenyp:%d, lenpt:%d' % (len(U), len(S), len(y_pseudo), len(pt)))
        
        error_k, error_val = perceptron_test(test, w_pred)
        #print('RS -iteration: %d, error: %f' % (k, error_k))
        test_errors_rs.append(error_k)
        test_errors_val_rs.append(error_val)

    #print(w_pred[0].item(), w_pred[1].item())
    #return test_errors_val_rs
    return pseudo_loss_rs, true_loss_rs, test_errors_val_rs

def main():
    train, test = generateData()

    n_exps = 10
    n_iterations = 1000

    
    #pseudo_loss_alis, true_loss_alis, _ = runALIS(train, test, n_iterations)
    #pseudo_loss_rs, true_loss_rs, _ = runRS(train, test, n_iterations)

    #print(pseudo_loss_alis)
    #print(true_loss_alis)
    #print(true_loss_rs, true_loss_alis)

    #plt.figure(1)
    #alis_tr, = plt.plot(true_loss_alis, label= 'ALIS_TR')
    #plt.legend(handles = [alis_tr])
    
    #plt.figure(2)
    #alis_ps, = plt.plot(pseudo_loss_alis, label= 'ALIS_PS')
    #plt.legend(handles = [alis_ps])

    #plt.figure(3)
    #one, = plt.plot(pseudo_loss_alis, label= 'one')
    #two, = plt.plot(true_loss_alis, label= 'two')
    #plt.legend(handles = [one, two])

    '''
    plt.figure(1)
    alis_ps, = plt.plot(pseudo_loss_alis, label= 'ALIS_PS')
    alis_tr, = plt.plot(true_loss_alis, label= 'ALIS_TR')
    rs_ps, = plt.plot(pseudo_loss_rs, label= 'RS_PS')
    rs_tr, = plt.plot(true_loss_rs, label= 'RS_TR')
    plt.ylabel('Average Test Set Error over %d experiments' % (n_exps))
    plt.xlabel('Iteration')
    plt.legend(handles=[alis_ps, alis_tr, rs_ps, rs_tr])
    plt.show()
    '''
    alis_avg_pseudoloss = [0]*n_iterations    
    alis_avg_trueloss = [0]*n_iterations
    alis_avg_test_errors = [0]*n_iterations

    rs_avg_pseudoloss = [0]*n_iterations
    rs_avg_trueloss = [0]*n_iterations
    rs_avg_test_errors = [0]*n_iterations

    for i in range(n_exps):
        print('***************EXPERIMENT %d *********' % i)
        alis_ps, alis_tr, alis_i = runALIS(train, test, n_iterations)
        alis_avg_pseudoloss = np.add(alis_avg_pseudoloss, np.divide(alis_ps, n_exps))
        alis_avg_trueloss = np.add(alis_avg_trueloss, np.divide(alis_tr, n_exps))
        alis_avg_test_errors = np.add(alis_avg_test_errors, np.divide(alis_i, n_exps))
        
        rs_ps, rs_tr, rs_i =  runRS(train, test, n_iterations)
        rs_avg_pseudoloss = np.add(rs_avg_pseudoloss, np.divide(rs_ps, n_exps))
        rs_avg_trueloss = np.add(rs_avg_trueloss, np.divide(rs_tr, n_exps))
        rs_avg_test_errors = np.add(rs_avg_test_errors, np.divide(rs_i, n_exps))


    #print(alis_avg_test_errors)
    #print(rs_avg_test_errors)

    plt.figure(1)
    alis_ps, = plt.plot(alis_avg_pseudoloss, label= 'ALIS_PS')
    alis_tr, = plt.plot(alis_avg_trueloss, label= 'ALIS_TR')
    rs_ps, = plt.plot(rs_avg_pseudoloss, label= 'RS_PS')
    rs_tr, = plt.plot(rs_avg_trueloss, label= 'RS_TR')
    plt.ylabel('Average Test Set Error over %d experiments' % (n_exps))
    plt.xlabel('Iteration')
    plt.legend(handles=[alis_ps, alis_tr, rs_ps, rs_tr])

    plt.figure(2)
    alis_ln, = plt.plot(alis_avg_test_errors, label= 'ALIS')
    rs_ln, = plt.plot(rs_avg_test_errors, label= 'RS')
    plt.ylabel('Average Test Set Error over %d experiments' % (n_exps))
    plt.xlabel('Iteration')
    plt.legend(handles=[alis_ln, rs_ln])
    plt.show()

    
    
if __name__ == '__main__':
    main()
    
    
