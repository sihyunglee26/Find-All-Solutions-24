'''
Authored by Sihyung Lee
To run the codes, the following modeuls must be installed:
    qiskit
    qiskit-aer
'''


from qiskit import QuantumCircuit, execute, Aer
import math
import random


def findAllSolutions(numQubits, numTargets, debug=False):
    '''
    This function simulates Step 1 (estimation) and Step 2 (discovery) of our proposed methods

    It randomly selects a subset of elements in the search space as solutions,
        runs the proposed methods to discover all solutions,
        and displays the results

    Input:
        numQubits - the number of qubits that represents the search space (n)
        numTargets - the number of targets within the search space (M)
    '''

    #
    # Randomly choose numTarget distinct targets
    #
    searchSpaceSize = 2 ** numQubits
    answersSet = generateAnswers(searchSpaceSize, numTargets)
    answersSortedList = sorted(list(answersSet)) 
    if debug: print(f"{numTargets} answers = {answersSortedList}")
    
    numDataQubits = numQubits    
    numAncillaQubits = numDataQubits - 1
    numMeasurements, numTotalIterations = 0, 0
    
    #
    # (Step 1) Estimate the number of targets
    #
    Mestimated, answersSetFound, numShots = estimateM(numQubits, answersSortedList)
    numMeasurements += numShots
    numTotalIterations += numShots
    if debug: print(f"numTargetsPredicted = {Mestimated}, where numTargetsLeft = {len(answersSet)}")

    #
    # (Step 2) Find all of the remaining targets
    #
    if Mestimated > 0: # Otherwise, if Mestimated == 0, then all solutions are found
        Mestimated = int(Mestimated + 0.5) - len(answersSetFound) # Use the closest integer as the estimate of M        
        for answerFound in answersSetFound: # remove answers found while estimating M
            answersSet.remove(answerFound)
            answersSortedList.remove(answerFound)
        
        numMeasurementsWithNoSolutions = 0        
        while numMeasurementsWithNoSolutions < 10:
            Mestimated = max(Mestimated, 1) # if Mestimated = 0, use 1 instead

            # If the oracle has changed, create a new quantum circuit (qc). Otherwise, use the same qc and save time
            if numMeasurementsWithNoSolutions == 0:
                qc = QuantumCircuit(numDataQubits + numAncillaQubits, numDataQubits) # numbers of qubits and classical bits            
                for qubit in range(numDataQubits): qc.h(qubit)  # initialization: set all data qubits into an even superposition, while ancilla qubits remain in state 0
                dataQubitIndices = [i for i in range(numDataQubits)]
                ancillaQubitIndices = [numDataQubits + i for i in range(numAncillaQubits)]
                numIterations = math.floor(math.sqrt(searchSpaceSize / Mestimated) * math.pi / 4)
                addControlledGroverIterator(qc, None, dataQubitIndices, ancillaQubitIndices, answersSortedList, numIterations)
                qc.measure(dataQubitIndices, range(numDataQubits))

            # Run the quantum circuit and store the results in measuredValue
            result = execute(qc, Aer.get_backend('qasm_simulator'), shots=1).result().get_counts(qc)
            numMeasurements += 1
            numTotalIterations += numIterations
            measuredValue = int(list(result.keys())[0], 2)
            if debug: print(f"measured value: {result} {measuredValue}")
            
            if measuredValue in answersSet: # If the measured value belongs to the solutions                
                if debug: print(f"a solution found and thus remove from answers")
                answersSet.remove(measuredValue)
                answersSortedList.remove(measuredValue)
                Mestimated -= 1
                numMeasurementsWithNoSolutions = 0
            else: # If the measured value does not belong to the solutions
                numMeasurementsWithNoSolutions += 1                

    print(f"Algorithm terminated with {numTargets - len(answersSet)}/{numTargets} solutions found")
    print(f"    performed {numMeasurements} measurements and {numTotalIterations} Grover iterations")
    print()


def generateAnswers(searchSpaceSize, numTargets):
    '''
    This function is used within findAllSolutions()
        to randomly generate numTargets solutions within a search space of size searchSpaceSize
    '''    
    answersSet = set() 
    for _ in range(numTargets):
        r = random.randint(0, searchSpaceSize - 1)
        while r in answersSet:
            r = random.randint(0, searchSpaceSize - 1)
        answersSet.add(r)
    return answersSet


def addControlledGroverIterator(qc, controlQubitIndex, dataQubitIndices, ancillaQubitIndices, answers, numIterations):
    '''
    This function is used within findAllSolutions()
        to create a quantum circuit that functions as a Grover's iterator
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


def estimateM(numQubits, answers, debug=False):
    '''
    This function is used within findAllSolutions()
        to simulate Step 1 (estimation)

    It creates an oracle from the list of solutions and
        runs the proposed methods to estimate the number of solutions
    
    Input:
        numQubits - the number of qubits that represents the search space (n)
        answers - a list of solutions (targets) within the search space

    Output:
        a 3-tuple containing
            the estimated number of solutions,
            the set of found solutions,
            and the number of Grover's iterations used in the estimation (for debugging)
    '''
    numDataQubits = numQubits    
    numAncillaQubits = numDataQubits - 1
    qc = QuantumCircuit(numDataQubits + numAncillaQubits, numDataQubits) # numbers of qubits and classical bits

    #
    # Initialization: set all data qubits into an even superposition
	#                   the ancilla qubits remain as they are (i.e., state 0)    
    dataQubitIndices = [i for i in range(numDataQubits)]
    ancillaQubitIndices = [numDataQubits + i for i in range(numAncillaQubits)]
    for qubit in dataQubitIndices: qc.h(qubit)    

    searchSpaceSize = 2 ** numQubits        
    numShots = int(math.sqrt(searchSpaceSize) * 10)    
    numIterations = 1

    addControlledGroverIterator(qc, None, dataQubitIndices, ancillaQubitIndices, answers, numIterations)
    qc.measure(dataQubitIndices, range(numDataQubits))    

	# run simulations and get results	
    result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numShots).result().get_counts(qc)    

    numSolutionsSampled = 0
    answersSet = set(answers)
    answersSetFound = set()
    for key in result.keys():
        keyInt = int(key, 2)
        if keyInt in answersSet: 
            numSolutionsSampled += result.get(key)
            answersSetFound.add(keyInt)
    
    if debug: print(f"numSolutionsSampled/numSamples = {numSolutionsSampled}/{numShots}")

    Mestimated = searchSpaceSize * (math.asin(math.sqrt(numSolutionsSampled / numShots)) ** 2) / 9    
    Mestimated = max(Mestimated, len(answersSetFound))    
    if debug: print(f"Mestimated/Mreal = {Mestimated:.5f}/{len(answers)}")

    return Mestimated, answersSetFound, numShots


if __name__ == "__main__":
    #
    # Run findAllSolutions() for various N and M
    #
    for n in range(3, 9+1):
        N = 2**n
        for M in range(0, math.floor(math.sqrt(N))+1):        
            print(f"N={N}, M={M}")
            findAllSolutions(n, M, False)
    