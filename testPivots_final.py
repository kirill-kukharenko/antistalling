import numpy as np
import random
from math import floor
from tqdm import tqdm
from cylp.py.pivots.PivotPythonBase import PivotPythonBase
from cylp.cy.CyClpSimplex import VarStatus
from cylp.cy import CyCoinIndexedVector
from cylp.cy.CyClpSimplex import cydot


itcount = 0
degenitcount = 0
samepoint = 0
DanzSamepoint = 0

class Degencounter:
    def __init__(self):
        self.counter = 0
        self.lastPoint = np.zeros((10000))
        self.tol = 1.e-7

    def setCounter(self, solLength, tol):
        self.counter = 0
        self.lastPoint = np.zeros((solLength))
        self.tol = tol

    def iterateCounter(self, curSolution):
        if np.allclose(curSolution, self.lastPoint, atol=self.tol, equal_nan=True):
            self.counter +=1
        self.lastPoint = np.array(curSolution)

degenCounter = Degencounter()

class AntistallingPivot(PivotPythonBase):
    '''
    Antistalling pivot rule implementation.
    '''

    def __init__(self, clpModel, optsol, optvalue):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel

        self.optsol = optsol
        self.optvalue = optvalue

        self.degenIter = False
        self.y_dir = np.zeros((clpModel.nRows + clpModel.nCols))
        
        self.lastLeavingIdx = -1
        self.lastEnteringIdx = -1
        self.lastBasis = np.zeros(clpModel.nCols)
        self.lastPoint = np.zeros((clpModel.nRows + clpModel.nCols))

        global degenCounter
        degenCounter.setCounter(clpModel.nRows + clpModel.nCols, clpModel.dualTolerance)


    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        s = self.clpModel
        global itcount
        itcount += 1
        
        # Update the reduced costs, for both the original and the slack variables
        if updates.nElements:
            s.updateColumnTranspose(spareRow2, updates)
            s.transposeTimes(-1, updates, spareCol2, spareCol1)
            s.reducedCosts[s.nVariables:][updates.indices] -= updates.elements[:updates.nElements]
            s.reducedCosts[:s.nVariables][spareCol1.indices] -= spareCol1.elements[:spareCol1.nElements]
        updates.clear()
        spareCol1.clear()

        rc = s.reducedCosts
        tol = s.dualTolerance

        x_cur = s.solution

        global degenCounter
        degenCounter.iterateCounter(x_cur)

        'comtuting new deriction y_dir after a non-degenarate pivot'
        if np.allclose(x_cur, self.lastPoint, atol=tol, equal_nan=True):
            global samepoint
            samepoint += 1
        else:
            self.y_dir = self.optsol - s.solution
        self.lastPoint = np.array(x_cur)


        objValue = np.dot(s.objective, s.solution[:s.nVariables])
        
        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0] 
        
        indicesToConsider1 = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree) & (np.abs(self.y_dir)>tol))[0] 

        if ((np.abs(objValue - self.optvalue) > tol) & (rc[indicesToConsider1].shape[0]>0)): 
            indicesToConsider = indicesToConsider1

        rc2 = np.abs(rc[indicesToConsider])

        checkFree = False
        enteringIdx = 0
        #rc2[np.where((status & 7 == 4) | (status & 7 == 0))] *= 10
        if rc2.shape[0] > 0:
            if checkFree:
                w = np.where(s.varIsFree)[0]
                if w.shape[0] > 0:
                    ind = s.argWeightedMax(rc2, indicesToConsider, 1, w)
                else:
                    ind = np.argmax(rc2)
            else:
                    ind = np.argmax(rc2)
            enteringIdx = indicesToConsider[ind]
        else: 
            return -1

        basis = np.where(s.varIsBasic)[0]

        lower_bounds = np.concatenate((s.variablesLower,s.constraintsLower))
        upper_bounds = np.concatenate((s.variablesUpper,s.constraintsUpper))

        'computing edge direcion'
        z = np.zeros((len(basis)))
        s.getBInvACol(enteringIdx, z)
        z_dir = np.zeros((s.nRows+s.nCols))
        z_dir[enteringIdx] = 1
        z_dir[basis] = -z

        degenIdx = np.where(s.varIsBasic & (np.invert(np.isclose(upper_bounds, lower_bounds, atol=tol, equal_nan=True))) &
                            (((np.isclose(x_cur, lower_bounds, atol=tol, equal_nan=True)) 
                              & (z_dir < -tol)) 
                              | ((np.isclose(x_cur, upper_bounds, atol=tol, equal_nan=True)) 
                                                      & (z_dir > tol))))[0]

        if len(degenIdx) > 0:
            'if the pivot is denenerate'
            global degenitcount
            degenitcount += 1
            self.degenIter = True

            'computing leaving index according to the antistalling rule'
            leavingIdx = np.argmin(np.abs(self.y_dir[degenIdx])/np.abs(z_dir[degenIdx]))
            alpha = np.min(np.abs(self.y_dir[degenIdx])/np.abs(z_dir[degenIdx]))  

            genLeavingIdx = degenIdx[leavingIdx]
            
            basisLeavingIdx = np.where(basis == genLeavingIdx)[0][0]
            
            'changing the direction y_dir according to the antistalling rule'
            self.y_dir = self.y_dir + alpha*z_dir

            s.setPivotRow(basisLeavingIdx)
            self.lastLeavingIdx = basisLeavingIdx
        else:
            self.degenIter = False
            self.lastLeavingIdx = -2

        self.lastEnteringIdx =  enteringIdx
        return enteringIdx
    

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        s = self.clpModel
        
        if self.degenIter:
            s.setPivotRow(self.lastLeavingIdx)
        return True
    
class SteepestEdgePivotDegen(PivotPythonBase):
    '''
    Steepest edge pivot rule implementation.
    '''
    def __init__(self, clpModel):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel

        global degenCounter
        degenCounter.setCounter(clpModel.nRows + clpModel.nCols, clpModel.dualTolerance)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        s = self.clpModel

        # Update the reduced costs, for both the original and the slack variables
        if updates.nElements:
            s.updateColumnTranspose(spareRow2, updates)
            s.transposeTimes(-1, updates, spareCol2, spareCol1)
            s.reducedCosts[s.nVariables:][updates.indices] -= updates.elements[:updates.nElements]
            s.reducedCosts[:s.nVariables][spareCol1.indices] -= spareCol1.elements[:spareCol1.nElements]
        updates.clear()
        spareCol1.clear()

        rc = s.reducedCosts
        tol = s.dualTolerance

        x_cur = s.solution
        global degenCounter
        degenCounter.iterateCounter(x_cur)

        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0]

        #freeVarInds = np.where(s.varIsFree)
        #rc[freeVarInds] *= 10

        rc2 = np.abs(rc[indicesToConsider])

        basis = np.where(s.varIsBasic)[0]
        
        for i in indicesToConsider: 
            'divides the corresponding reduced cost (= edge direction times objective vector) by the 2 norm of the edge direction'
            z = np.zeros((s.nRows))
            s.getBInvACol(i, z)

            z_norm_2 = np.sqrt(np.linalg.norm(z, 2)**2+1) 

            idx_rc2 = np.where(rc2 == np.abs(rc[i]))

            rc2[idx_rc2] /= z_norm_2
            del z
            del idx_rc2 


        checkFree = False
        #rc2[np.where((status & 7 == 4) | (status & 7 == 0))] *= 10
        if rc2.shape[0] > 0:
            if checkFree:
                w = np.where(s.varIsFree)[0]
                if w.shape[0] > 0:
                    ind = s.argWeightedMax(rc2, indicesToConsider, 1, w)
                else:
                    ind = np.argmax(rc2)
            else:
                    ind = np.argmax(rc2)
            del rc2
            return  indicesToConsider[ind]
        return -1

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        return True

class DantzigPivotDegen(PivotPythonBase):
    '''
    Dantzig's pivot rule implementation.
    Adapted from: https://coin-or.github.io/CyLP/_modules/cylp/py/pivots/DantzigPivot.html#DantzigPivot
    Copyright 2011, Mehdi Towhidi, Dominique Orban. Created using Sphinx 1.6.7.
    '''

    def __init__(self, clpModel):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel

        global degenCounter
        degenCounter.setCounter(clpModel.nRows + clpModel.nCols, clpModel.dualTolerance)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        s = self.clpModel

        # Update the reduced costs, for both the original and the slack variables
        if updates.nElements:
            s.updateColumnTranspose(spareRow2, updates)
            s.transposeTimes(-1, updates, spareCol2, spareCol1)
            s.reducedCosts[s.nVariables:][updates.indices] -= updates.elements[:updates.nElements]
            s.reducedCosts[:s.nVariables][spareCol1.indices] -= spareCol1.elements[:spareCol1.nElements]
        updates.clear()
        spareCol1.clear()

        rc = s.reducedCosts
        tol = s.dualTolerance

        x_cur = s.solution
        global degenCounter
        degenCounter.iterateCounter(x_cur)

        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0]

        #freeVarInds = np.where(s.varIsFree)
        #rc[freeVarInds] *= 10

        rc2 = np.abs(rc[indicesToConsider])

        checkFree = True
        #rc2[np.where((status & 7 == 4) | (status & 7 == 0))] *= 10
        if rc2.shape[0] > 0:
            if checkFree:
                w = np.where(s.varIsFree)[0]
                if w.shape[0] > 0:
                    ind = s.argWeightedMax(rc2, indicesToConsider, 1, w)
                else:
                    ind = np.argmax(rc2)
            else:
                    ind = np.argmax(rc2)
            #del rc2
            return  indicesToConsider[ind]
        return -1

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        return True
    
class LIFOPivotDegen(PivotPythonBase):
    '''
    Last-In-First-Out pivot rule implementation. Turns to Bland's rule is the argument isBland is True
    Adapted from: https://coin-or.github.io/CyLP/_modules/cylp/py/pivots/LIFOPivot.html#LIFOPivot
    Copyright 2011, Mehdi Towhidi, Dominique Orban. Created using Sphinx 1.6.7.
    '''
    def __init__(self, clpModel, isBland=False):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel
        #self.banList = np.zeros(self.dim, np.int)
        self.banList = []
        self.priorityList = list(range(self.dim))

        self.isBland = isBland
        global degenCounter
        degenCounter.setCounter(clpModel.nRows + clpModel.nCols, clpModel.dualTolerance)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel
        rc = s.reducedCosts
        dim = s.nCols + s.nRows

        tol = s.dualTolerance

        x_cur = s.solution
        global degenCounter
        degenCounter.iterateCounter(x_cur)

        for i in self.priorityList:
            #flagged or fixed
            if s.flagged(i) or s.CLP_getVarStatus(i) == VarStatus.fixed:
                continue

            #TODO: can we just say dualInfeasibility = rc[i] ** 2
            if s.CLP_getVarStatus(i) == VarStatus.atUpperBound:  # upperbound
                dualInfeasibility = rc[i]
            elif (s.CLP_getVarStatus(i) == VarStatus.superBasic or
                    s.CLP_getVarStatus(i) == VarStatus.free):  # free or superbasic
                dualInfeasibility = abs(rc[i])
            else:  # lowerbound
                dualInfeasibility = -rc[i]

            if dualInfeasibility > tol:
                return i

        return -1

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        if self.isBland:
            return True
        '''
        Inserts the leaving variable index as the first element
        in self.priorityList
        '''
        s = self.clpModel

        pivotRow = s.pivotRow()
        if pivotRow < 0:
            return True

        pivotVariable = s.getPivotVariable()
        leavingVarIndex = pivotVariable[pivotRow]

        self.priorityList.remove(leavingVarIndex)
        self.priorityList.insert(0, leavingVarIndex)

        return True
    
class PositiveEdgePivotDegen(PivotPythonBase):
    '''
    Positive Edge pivot rule implementation.
    Adapted from: https://coin-or.github.io/CyLP/_modules/cylp/py/pivots/PositiveEdgePivot.html#PositiveEdgePivot
    Copyright 2011, Mehdi Towhidi, Dominique Orban. Created using Sphinx 1.6.7.
    '''

    def __init__(self, clpModel, EPSILON=10 ** (-7)):
        self.clpModel = clpModel
        self.dim = self.clpModel.nRows + self.clpModel.nCols

        self.isDegenerate = False

        # Create some numpy arrays here ONCE to prevent memory
        # allocation at each iteration
        self.aColumn = CyCoinIndexedVector()
        self.aColumn.reserve(self.dim)
        self.w = CyCoinIndexedVector()
        self.w.reserve(self.clpModel.nRows)

        self.rhs = np.empty(self.clpModel.nRows, dtype=np.double)
        self.EPSILON = EPSILON
        self.lastUpdateIteration = 0

        global degenCounter
        degenCounter.setCounter(clpModel.nRows + clpModel.nCols, clpModel.dualTolerance)

    def updateP(self):
        '''Finds constraints with abs(rhs) <=  epsilon and put
        their indices in "z"
        '''
        s = self.clpModel
        nRows = s.nRows

        rhs = self.rhs
        s.getRightHandSide(rhs)

        #self.p = np.where(np.abs(rhs) > self.EPSILON)[0]
        self.z = np.where(np.abs(rhs) <= self.EPSILON)[0]

        #print 'degeneracy level : ', (len(self.z)) / float(nRows)
        self.isDegenerate = (len(self.z) > 0)

    def updateW(self):
        '''Sets "w" to be a vector of random vars with "0"
        at indices defined in "p"
        Note that vectorTimesB_1 changes "w"
        '''
        self.updateP()
        self.w.clear()
        self.w[self.z] = np.random.random(len(self.z))
        s = self.clpModel
        s.vectorTimesB_1(self.w)

        self.lastUpdateIteration = s.iteration

    def random(self):
        'Defines how random vector "w" components are generated'
        return random.random()

    def isCompatible(self, varInd):
        if not self.isDegenerate:
            return False
        s = self.clpModel
        s.getACol(varInd, self.aColumn)

        return abs(cydot(self.aColumn, self.w)) < self.EPSILON

    def checkVar(self, i):
        return self.isCompatible(i)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel
        rc = s.reducedCosts

        tol = s.dualTolerance

        x_cur = s.solution
        global degenCounter
        degenCounter.iterateCounter(x_cur)

        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0]

        rc2 = abs(rc[indicesToConsider])

        maxRc = maxCompRc = maxInd = maxCompInd = -1

        if self.isDegenerate:
            w = self.w.elements
            compatibility = np.zeros(s.nCols + s.nRows, dtype=np.double)
            if len(indicesToConsider) > 0:
                s.transposeTimesSubsetAll(indicesToConsider,
                                          w, compatibility)
            comp_varInds = indicesToConsider[np.where(abs(
                                    compatibility[indicesToConsider]) <
                                    self.EPSILON)[0]]

            comp_rc = abs(rc[comp_varInds])
            if len(comp_rc) > 0:
                maxCompInd = comp_varInds[np.argmax(comp_rc)]
                maxCompRc = rc[maxCompInd]

        if len(rc2) > 0:
            maxInd = indicesToConsider[np.argmax(rc2)]
            maxRc = rc[maxInd]

        del rc2
        if maxCompInd != -1 and abs(maxCompRc) > 0.1 * abs(maxRc):
            return maxCompInd
        self.updateW()
        return maxInd

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        return True
    
class MostFrequentPivotDegen(PivotPythonBase):
    '''
    Most frequents pivot rule implementation.
    Adapted from: https://coin-or.github.io/CyLP/_modules/cylp/py/pivots/MostFrequentPivot.html#MostFrequentPivot
    Copyright 2011, Mehdi Towhidi, Dominique Orban. Created using Sphinx 1.6.7.
    '''

    def __init__(self, clpModel):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel
        #self.banList = np.zeros(self.dim, np.int)
        self.banList = []
        self.priorityList = list(range(self.dim))
        self.frequencies = np.zeros(self.dim)

        global degenCounter
        degenCounter.setCounter(clpModel.nRows + clpModel.nCols, clpModel.dualTolerance)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel
        rc = s.getReducedCosts()
        dim = s.nRows + s.nCols

        tol = s.dualTolerance

        x_cur = s.solution
        global degenCounter
        degenCounter.iterateCounter(x_cur)

        for i in self.priorityList:
            if s.flagged(i) or s.CLP_getVarStatus(i) == 5:  # flagged or fixed
                continue

            #TODO: can we just say dualInfeasibility = rc[i] ** 2
            if s.CLP_getVarStatus(i) == 2:  # upperbound
                dualInfeasibility = rc[i]
            # free or superbasic
            elif s.CLP_getVarStatus(i) == 4 or s.CLP_getVarStatus(i) == 0:
                dualInfeasibility = abs(rc[i])
            else:  # lowerbound
                dualInfeasibility = -rc[i]

            if dualInfeasibility > tol:
                self.addFrequency(i)
                return i

        return -1

    def addFrequency(self, i):
        '''
        Add one to frequency of variable i,
        resorts the priorityList (always sorted)
        '''
        self.frequencies[i] += 1
        self.priorityList.remove(i)
        for j in range(self.dim):
            if self.frequencies[i] >= self.frequencies[self.priorityList[j]]:
                self.priorityList.insert(j, i)
                return
        self.priorityList.append(i)

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        return True

   
if __name__ == "__main__":
    'here the relative path to your folder with .mps problem files'
    foldername = "test_probs"

    outputFile = "test.csv"

    from cylp.cy import CyClpSimplex
    import csv
    import os
    import signal

    def handler(signum, frame):
        raise Exception("timelim")
    
    def applPivot(pivotFun, lpfile, pert, *args):
        'function to apply simplex with specified pivot pivotFun to the problem lpfile'
        s = CyClpSimplex()
        
        s.readMps(foldername+'/'+lpfile)
        print (pivotFun)

        pivot = pivotFun(s, *args)

        s.setPerturbation(pert)

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(1800)

        try:
            s.setPivotMethod(pivot)
            s.useCustomPrimal(True)
            s.primal() 

        except Exception:
            return ('*','*','*','*')
        signal.alarm(0)

        return (s.solution, s.objectiveValue, s.iteration, degenCounter.counter)
    
    'no perturabtion'
    pert = 102

    'all problems to consider'
    lpList = [filename for filename in os.listdir(os.getcwd()+ '/' +foldername)]
    lpList = sorted(lpList)
    print (lpList)

    csv_header = ['problem', 'Dantzig', 'PositiveEdge', 'LIFO', 'MostFrequent', 'SteepestEdge','Blands', 'Antistalling']  
    'all results will be saved in this array'
    csv_file = []

    idx = 0
    for lpfile in tqdm(lpList):
        listPivots =  [DantzigPivotDegen, PositiveEdgePivotDegen, LIFOPivotDegen, MostFrequentPivotDegen, SteepestEdgePivotDegen] 
        print ("### Working on "+ lpfile+" ###")
        csv_file.append([lpfile])
        csv_file.append([lpfile+'.degen'])

        listResults =  []
        for pivot in listPivots:
            listResults.append(applPivot(pivot, lpfile, pert))

        listPivots.append("BlandsPivotDegen")
        listResults.append(applPivot(LIFOPivotDegen, lpfile, pert, True))

        listPivots.append(AntistallingPivot)
        if len(listPivots) > 0 and (listResults[0][2] != '*'):   
            listResults.append(applPivot(AntistallingPivot, lpfile, pert, listResults[0][0], listResults[0][1]))
        else:
            listResults.append(('*','*','*','*'))

        for i in range(len(listPivots)):
            print (listPivots[i], "#iter", listResults[i][2], "#degenIter", listResults[i][3])
            csv_file[2*idx].append(listResults[i][2])
            csv_file[2*idx+1].append(listResults[i][3])
        
        # writing to csv file
        with open(outputFile, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_header)

            csvwriter.writerows(csv_file)

        idx += 1