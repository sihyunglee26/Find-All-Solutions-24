'''
Authored by Sihyung Lee
To run the codes, the following modeuls must be installed:
    qiskit
    qiskit-aer

When writing this code, we referred to the following page:
    https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/quantum-counting.ipynb
'''


from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.library import QFT
import math
import random


def quantumCounting(numQubits, answers, numShots=100):
    '''
    This function simulates Quantum Counting for estimating the number of solutions in a search space

    Input:
        numQubits - the number of qubits that represents the search space (n)
        answers - a list of solutions (targets) within the search space

    Output:
        a 4-tuple containing
            the number of counting qubits (t) used in the estimation,
            the estimated number of solutions (Mhat),
            minimum value of M, considering the range of errors,
            and maximum value of M, considering the range of errors    
    '''

    #
    # Begin creating a quantum circuit
    #
    numCountingQubits = math.ceil(math.log(math.sqrt(2 ** numQubits) + 1, 2))   # # of counting qubits (t) as per Section IV.A
    numDataQubits = numQubits    
    numAncillaQubits = numDataQubits - 1
    qc = QuantumCircuit(numCountingQubits + numDataQubits + numAncillaQubits, numCountingQubits) # numbers of qubits and classical bits
    
    #
    # Initialization: set all counting/data qubits into an even superposition
	#                   the ancilla qubits remain as they are (i.e., state 0)
    #
    for qubit in range(numCountingQubits + numDataQubits):
        qc.h(qubit)
    searchSpaceSize = 2 ** numQubits
    countingQubitIndices = [i for i in range(numCountingQubits)]
    dataQubitIndices = [numCountingQubits + i for i in range(numDataQubits)]
    ancillaQubitIndices = [numCountingQubits + numDataQubits + i for i in range(numAncillaQubits)]    

    #
    # Add 2^t - 1 Grover's iterations to the quantum circuit
    #
    numIterations = 1
    for controlQubitIndex in countingQubitIndices:
        # (method 2) Use Grover's operator that we implemented
        addControlledGroverIterator(qc, controlQubitIndex, dataQubitIndices, ancillaQubitIndices, answers, numIterations)
        numIterations *= 2    

    # Add an inverse QFT
    qc.append(qftInverse(numCountingQubits), countingQubitIndices)

    #
    # Error correction - using H^n -D H^n instead of H^n D H^n 
    #   in the 1st application of Grover's iterator (with 2^0 = 1 iteration)
    #   leads to a flipped probability distribution for the last counting bit
    #   and thus we flip it back before measurement
    #
    qc.x(countingQubitIndices[-1]) 

    #
    # Measure the counting qubits and store the measurements in the classical bits
    #
    qc.measure(countingQubitIndices, range(numCountingQubits))    

    #
	# Run simulations and get results	
    #
    result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numShots).result().get_counts(qc)    

    # Estimate the number of solutions (M) from the results
    maxkey = max(result, key=result.get)
    measuredInt = int(maxkey, 2) # take the measurement with max frequency as int
    theta = (measuredInt / (2**numCountingQubits)) * math.pi * 2
    numTargetsPredicted = searchSpaceSize * (math.sin(theta/2)**2)
    error = (math.sqrt(2*numTargetsPredicted*searchSpaceSize) + searchSpaceSize/(2**numCountingQubits)) * (2**(1-numCountingQubits))

    # compute the min/max integer values of M, considering the error range
    minM = max(math.ceil(numTargetsPredicted - error), 0.0) # M >= 0
    maxM = min(math.floor(numTargetsPredicted + error), N) # M <= N
    if minM > maxM: minM, maxM = maxM, minM # Swap min and max integer if the range contains no integers    
    
    return numCountingQubits, numTargetsPredicted, minM, maxM

    
def qftInverse(numQubits):
    '''
    This function is used within quantumCounting()
        to create a quantum circuit that functions as an inverse QFT
    '''
    result = QFT(numQubits, inverse=True).to_gate() # return as a Gate instance    
    return result


def addControlledGroverIterator(qc, controlQubitIndex, dataQubitIndices, ancillaQubitIndices, answers, numIterations):
    '''
    This function is used within quantumCounting()
        to create a quantum circuit that functions as a Grover's iterator

    We created our own Grover's circuit, rather than auto-generating it from Qiskit,
        since our own circuit is simpler and faster
    '''
    #
    # Sanity check
    #
    assert isinstance(qc, QuantumCircuit), f"qc of class {type(qc).__name__} must be a QuantumCircuit instance"
    assert isinstance(controlQubitIndex, int) or controlQubitIndex == None, f"controlQubitIndex(={controlQubitIndex}) must be an integer or None"
    assert isinstance(dataQubitIndices, list) and all(isinstance(e, int) for e in dataQubitIndices), f"dataQubitIndices(={dataQubitIndices} must be a list with integers)"
    numDataQubits = len(dataQubitIndices)
    assert numDataQubits >= 2, f"The number of data qubits(={numDataQubits}) must be >=2 for Grover's algorithm to be effective in searching a target"
    assert isinstance(ancillaQubitIndices, list) and all(isinstance(e, int) for e in ancillaQubitIndices), f"ancillaQubitIndices(={ancillaQubitIndices} must be a list with integers)"        
    numAncillaQubits = len(ancillaQubitIndices)
    assert numAncillaQubits == numDataQubits - 1 , f"The number of ancilla qubits(={numDataQubits}) must be equal to the number of ancilla qubits(={numAncillaQubits} minus one)"
    searchSpaceSize = 2 ** numDataQubits
    assert isinstance(answers, list) and all(isinstance(e, int) for e in answers) and all(0<=e and e<=searchSpaceSize-1 for e in answers), f"answers(={answers} must be a list with integers within [0, {searchSpaceSize-1}])"
    assert isinstance(numIterations, int), f"numIterations(={numIterations}) must be an integer"
    answersInBinary = [f"{answer:0{numDataQubits}b}" for answer in answers]    

    for _ in range(numIterations):
        #
        # Oracle - sign-invert each answer
        #
        for answerInBinary in answersInBinary:
            for bitPosition in range(numDataQubits):
                if answerInBinary[numDataQubits - bitPosition - 1] == '0': qc.x(dataQubitIndices[bitPosition])                
        
            qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])
            for i in range(2, numDataQubits):
                qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])
                        
            if controlQubitIndex == None: qc.cz(ancillaQubitIndices[-1], dataQubitIndices[-1])
            else: qc.ccz(controlQubitIndex, ancillaQubitIndices[-1], dataQubitIndices[-1])  # controlled sign-invert
            
            for i in reversed(range(2, numDataQubits)):
                qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])
            qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])

            for bitPosition in range(numDataQubits):
                if answerInBinary[numDataQubits - bitPosition - 1] == '0': qc.x(dataQubitIndices[bitPosition])            

        #
        # Flip about the mean (amplitude amplification)
        #        
        for qubit in dataQubitIndices: qc.h(qubit)
        for qubit in dataQubitIndices: qc.x(qubit)

        qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])
        for i in range(2, numDataQubits):
            qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])
        
        if controlQubitIndex == None: qc.cz(ancillaQubitIndices[-1], dataQubitIndices[-1])
        else: qc.ccz(controlQubitIndex, ancillaQubitIndices[-1], dataQubitIndices[-1])  # controlled sign-invert

        for i in reversed(range(2, numDataQubits)):
            qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])
        qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])

        for qubit in dataQubitIndices: qc.x(qubit)
        for qubit in dataQubitIndices: qc.h(qubit)


def generateAnswers(searchSpaceSize, numTargets):
    '''
    This function is used in "__main__"
        to randomly generate numTargets solutions within a search space of size searchSpaceSize
    '''  
    answersSet = set() 
    for _ in range(numTargets):
        r = random.randint(0, searchSpaceSize - 1)
        while r in answersSet:
            r = random.randint(0, searchSpaceSize - 1)
        answersSet.add(r)
    return answersSet


if __name__ == "__main__":
    #
    # Run quantumCounting() for various N and M
    #   and print average error
    #
    for n in range(3, 6+1):    
        N = 2**n        
        for M in range(0, math.floor(math.sqrt(N))+1):        
            print(f"N={N}, M={M}: ", end='')
            answersSet = generateAnswers(N, M)
            answersSortedList = sorted(list(answersSet))
            errorList = []
            for _ in range(20):
                numCountingQubits, numTargetsPredicted, _, _ = quantumCounting(n, answersSortedList, numShots=10)  # numShot = 10 as per Section IV.A
                errorList.append(abs(M - numTargetsPredicted))
            print(f"avg. error = {sum(errorList) / len(errorList):.4f} with {numCountingQubits} counting qubits")
