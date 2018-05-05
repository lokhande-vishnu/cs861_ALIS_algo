import numpy as np
import matplotlib.pyplot as plt

trainsize = 5000
testsize = 10000
label_fraction = 0.001
h_star = [0.5, -0.5]
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
    xtrain = np.random.uniform(-1, 1, (trainsize,2))
    train = []
    for i in range(len(xtrain)):
        if np.dot(xtrain[i], h_star) >= 0:
            train.append([xtrain[i], 1])
        else:
            train.append([xtrain[i], -1])

    xtest = np.random.uniform(-1, 1, (testsize,2))
    test = []
    for i in range(len(xtest)):
        if np.dot(xtest[i], h_star) >= 0:
            test.append([xtest[i], 1])
        else:
            test.append([xtest[i], -1])

    return train, test

def breakData(train):
    S = []
    U = []
    for i in range(len(train)):
        if i*1.0/len(train) < label_fraction:
            S.append([train[i][0], train[i][1]])
        else:
            U.append([train[i][0], train[i][1]])
    return S, U

def perceptron_train(ltrain, w_init, ltrain_pt, eta_):
    w = w_init
    eta = eta_
    epochs = 1

    print('training', sum(ltrain_pt))
    for t in range(epochs):
        for idx, (x, label) in enumerate(ltrain):
            if (np.dot(x, w)*label) < 0:
                w += 1.0*eta*x*label/ltrain_pt[idx]
                print('CORRECTION', 1.0*eta*x*label/ltrain_pt[idx])
    return w

def perceptron_test(data, w):
    error = 0
    for x, label in data:
        if (np.dot(x, w)*label) < 0:
            error += np.dot(x,w)*label # Hinge loss
    return -error

def compute_y_pseudo(U, w_pred):
    y_pseudo = []
    for i in range(len(U)):
        if np.dot(U[i][0], w_pred) >= 0:
            y_pseudo.append(-1)
        else:
            y_pseudo.append(1)
    return y_pseudo

def psuedo_loss(U, pt, y_pseudo, w_pred):
    t = 0
    for i in range(len(U)):
        try:
            t += 1.0*np.square(max(1-y_pseudo[i]*np.dot(w_pred, U[i][0]), 0)) / pt[i]
        except:
            print(i)
            print(len(y_pseudo))
            print(len(U))
            print(len(pt))
            exit(1)
    return t
        

def compute_pt(U, w_pred, y_pseudo):
    l = []
    for i in range(len(U)):
        #print(w_pred, 1-y_pseudo[i]*np.dot(U[i][0], w_pred))
        l.append(max(1-y_pseudo[i]*np.dot(U[i][0], w_pred), 0)) # Hinge loss
    
    pt = [x / sum(l) for x in l]

    print('sumn of pt\n', sum(pt))
    print('\n')
    return pt

def sample(U, pt):
    index_V = set(np.random.choice(np.arange(len(U)), m, pt))
    V = []
    V_pt = []
    for idx in index_V:
        V.append(U[idx])
        V_pt.append(pt[idx])
    return V, index_V, V_pt

def update_sets(U, S, index_V):
    for idx in sorted(index_V, reverse= True):
        try:
            S.append(U[idx])
            del U[idx]
        except:
            print('....ERROR....')
            print(idx)
            print(index_V)
            print(len(U))
    return U, S

def main():
    train, test = generateData()


    # ALIS algorithm
    S, U = breakData(train)

    print('\n Running  ALIS algorithm ...')
    print('Size of labelled dataset:%d, Size of unlabelled dataset:%d' % (len(S), len(U)))

    eta_ = 1e-5
    pt_init = [(1.0/len(S))]*len(S)
    w_init = np.zeros(2)
    w_pred = perceptron_train(S, w_init, pt_init, eta_)
    print(perceptron_test(S, w_pred))

    test_errors_alis = []
    psuedo_loss_alis = []
    for k in range(450):
        y_pseudo = compute_y_pseudo(U, w_pred)
        pt = compute_pt(U, w_pred, y_pseudo)

        print('before', len(U), len(S), len(y_pseudo), len(pt))
        V, index_V, V_pt =  sample(U, pt)
        print('w_pred_before', w_pred)
        w_pred = perceptron_train(V, w_pred, V_pt, eta_)
        #psuedo_loss_alis.append(psuedo_loss(U, pt, y_pseudo, w_pred))
        U, S = update_sets(U, S, index_V)
        print('w_pred_after', w_pred)
        print('after', len(U), len(S))
        
        error_k = perceptron_test(test, w_pred)

        print('ALIS - iteration: %d, error: %f' % (k, error_k))
        test_errors_alis.append(error_k)

    # Random Sampling algorithm
    S, U = breakData(train)
    print('Running  random sampling algorithm ...')
    print('Size of labelled dataset:%d, Size of unlabelled dataset:%d' % (len(S), len(U)))

    #eta_ = 0.00000000001
    eta_ = 1e-1
    pt_init = [1.0/len(S)]*len(S)
    w_init = np.zeros(2)
    w_pred = perceptron_train(S, w_init, pt_init, eta_)
    print(perceptron_test(S, w_pred))

    test_errors_rs = []
    psuedo_loss_rs = []
    for k in range(450):
        y_pseudo = compute_y_pseudo(U, w_pred)
        pt = [1.0/len(U)]*len(U)
        print('before', len(U), len(S))
        V, index_V, _ =  sample(U, pt)
        V_pt = [1]*len(V)
        print('w_pred_before', w_pred)
        w_pred = perceptron_train(V, w_pred, V_pt, eta_)
        #psuedo_loss_rs.append(psuedo_loss(U, pt, y_pseudo, w_pred))
        U, S = update_sets(U, S, index_V)
        print('w_pred_after', w_pred)
        print('after', len(U), len(S))
        
        error_k = perceptron_test(test, w_pred)
        print('RS -iteration: %d, error: %f' % (k, error_k))
        test_errors_rs.append(error_k)


    alis_ln, = plt.plot(test_errors_alis, label= 'alis')
    rs_ln, = plt.plot(test_errors_rs, label= 'rs')
    plt.ylabel('Test Set Error')
    plt.xlabel('Iteration')
    plt.legend(handles=[alis_ln, rs_ln])
    plt.show()


if __name__ == '__main__':
    main()
    
    
