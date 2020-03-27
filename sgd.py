import numpy as np
import math
def SGDSolver(string,x, y, alpha, lam, nepoch, epsilon, param):
    x = x - x.mean(axis=0)
    x = x / np.abs(x).max(axis=0)
    
    if string == 'Training':
        class0Input = []
        class1Input = []
        class2Input = []
        zerovs12y = []
        onevs02y = []
        twovs01y = []
        for i in range(len(x)):
            if(y[i] >= 7):
                class2Input.append(x[i])
            elif(y[i] <= 4):
                class0Input.append(x[i])
            else:
                class1Input.append(x[i])
        #training 0 vs 12
        for i in range(len(x)):
            if(y[i] <= 4):
                zerovs12y.append(1)
            else:
                zerovs12y.append(0)
         #training 2 vs 01
        for i in range(len(x)):
            if(y[i] >= 7):
                twovs01y.append(1)
            else:
                twovs01y.append(0)
         #training 1 vs 02
        for i in range(len(x)):
            if(y[i] >= 7):
                onevs02y.append(0)
            elif(y[i] <= 4):
                onevs02y.append(0)
            else:
                onevs02y.append(1)

        

        answer = gridSearch(alpha, lam, x, zerovs12y, nepoch, param)
        alpha = answer[0]
        lam = answer[1]
        zerovs12params = []
        zerovs12params = trainParams(x, zerovs12y, alpha, lam, 300, zerovs12params)
        onevs02params = []
        onevs02params = trainParams(x, onevs02y, alpha, lam, 300, onevs02params)
        twovs01params = []
        twovs01params = trainParams(x, twovs01y, alpha, lam, 300, twovs01params)
        param = []
        param.append(zerovs12params)
        param.append(onevs02params)
        param.append(twovs01params)
            
        
        #return trainParams(x, y, alpha, lam, nepoch, param)
        ynew = []  
        for i in y:
            if(i >= 7):
                ynew.append(2)
            elif(i <= 4):
                ynew.append(0)
            else:
                ynew.append(1)
        
        index = 0
        correct = 0
        for row in x:
            if calculate(row, param[0]) > calculate(row, param[1]) and calculate(row, param[0]) > calculate(row, param[2]):
                yhat = 0
            elif calculate(row, param[1]) > calculate(row, param[0]) and calculate(row, param[1]) > calculate(row, param[2]):
                yhat = 1
            else:
                yhat = 2
            print(yhat, ynew[index], " ")
            if( yhat == ynew[index]):
                correct = correct + 1
            index = index +1
        acc = correct/index
        print(acc)
        return param, alpha, lam
        


    if string == 'Validation':
        return validation(x, y, param)
    if string == 'Testing':
        return testing(x, param)
        
def crossEnt(ypredicted, yactual):
    return -(yactual*np.log(ypredicted)+(1-yactual)*np.log(1-ypredicted))

def testing(x, param):
    index = 0
    yhat = []
    for row in x:
        if calculate(row, param[0]) > calculate(row, param[1]) and calculate(row, param[0]) > calculate(row, param[2]):
            yhat.append(0)
        elif calculate(row, param[1]) > calculate(row, param[0]) and calculate(row, param[1]) > calculate(row, param[2]):
            yhat.append(1)
        else:
            yhat.append(2)
            index = index +1
    return yhat

def ErrorCalculator(x, y, alpha, lam, nepoch, param):   
    param = [0.5 for p in range(len(x[0]))]
    errorTotal = 0
    for index in range(len(x)):
        prediction = calculate(x[index], param)
        error = crossEnt(prediction, y[index])
        errorTotal += error
        if(abs(errorTotal) > (2 ** 31 - 1)):
            break
        #print(errorTotal)
        for i in range(len(x[index])):
            param[i] = param[i] - alpha*error*x[index][i]
    return errorTotal
           
def gridSearch(alpha,lam,x,y,nepoch, param):
    bestError = 1000000000
    bestAlpha = 0
    bestLam = 0
    for a in np.arange(alpha[0], alpha[1], .05):
       for l in np.arange(lam[0], lam[1], .01):
            params = [0.5 for i in range(len(x[0]))]
            sumofCrossEntry = ErrorCalculator(x,y,a,l,nepoch,params)
            print(sumofCrossEntry)
            if bestError > sumofCrossEntry:
                bestAlpha = a
                bestLam = l
                bestError = sumofCrossEntry
    print(bestError, bestAlpha, bestLam, sep= " ")
    answer = [bestAlpha, bestLam]
    return answer

def trainParams(x, y, alpha, lam, nepoch, params):   
    params = [0.5 for i in range(len(x[0]))]
    for n in range(nepoch):
        errorTotal = 0
        count = 0
        for row in x:
            prediction = calculate(row, params)
            error = y[count] - prediction
            if(abs(errorTotal) > (2 ** 31 - 1)):
                break
            count += 1
            for i in range(len(row)):
                 params[i] = params[i] + alpha*error*row[i]*prediction
    return params

def calculate(row, param):
    calc = 0
    for x in range(len(row)):
        calc += param[x]*row[x]
    sigmoid = 1.0/(1.0+np.exp(0-calc))
    return sigmoid

def validation(x, y, param):
    sumError = 0
    count = 0
    for row in x:
        for i in param:
            prediction = calculate(row, i)
            error = prediction - y[count]
            print(prediction, y[count], " ")
        count += 1
        sumError = error*error
        
    print(math.sqrt(sumError/count))
    return math.sqrt(sumError/count)
    
    
