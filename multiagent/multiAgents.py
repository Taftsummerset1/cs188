# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys
from game import Agent
from game import Actions

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        
	newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
	foodList = newFood.asList()
	m = sys.maxint
	numFood = len(foodList) 
	
	#finding the closest food pellet
	for f in foodList:
	    d = util.manhattanDistance(newPos, f)
	    if d < m:
	        m = d
	#prevents m from being maxint if there is no food left
	if len(foodList) == 0:
	   m = 0
	food = m
	
	#finding the closest ghost
	g = sys.maxint
	for ghost in newGhostStates:
	    if newScaredTimes[newGhostStates.index(ghost)] == 0:
		d = util.manhattanDistance(newPos, ghost.getPosition())
		if d < g:
		    g = d
	
	#prevents g from being maxint if there are no ghosts
	if len(newGhostStates) == 0:
	    g = 0
	#if the ghost is 1 or less spaces away, BAD
	if g <= 1:
	    g = -10000000000000000
	
	#the evalution function to be returned
	r = successorGameState.getScore() - (1.0 * food) + (g/1.05) - (10.0 * numFood)
	return r

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.value(gameState, 0, 0)[1]

    def maxValue(self, state, agentIndex, level):
        v = float("-inf")
        best = "Stop"
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            scoreAction = self.value(successor, agentIndex + 1, level)
            if scoreAction[0] > v:
                v = scoreAction[0]
                best = action
        return (v, best)

    def minValue(self, state, agentIndex, level):
        v = float("inf")
        best = "Stop"

        if level == self.depth - 1 and agentIndex == (state.getNumAgents() - 1):
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                scoreAction = (self.evaluationFunction(successor), action)
                if scoreAction[0] < v:
                    v = scoreAction[0]
                    best = action
            return (v, best)
        else:
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                scoreAction = self.value(successor, agentIndex + 1, level)
                if scoreAction[0] < v:
                    v = scoreAction[0]
                    best = action
            return (v, best)

    def value(self, state, agentIndex, level):
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state), "Stop")
        elif agentIndex == state.getNumAgents(): #last ghost
            return self.value(state, 0, level + 1)
        elif agentIndex == 0: #pacman
            return self.maxValue(state, agentIndex, level)
        else: #if ghost 
            return self.minValue(state, agentIndex, level)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.value(gameState, 0, 0, float("-inf"), float("inf"))[1]

    def maxValue(self, state, agentIndex, level, alpha, beta):
        v = float("-inf")
        best = "Stop"
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            scoreAction = self.value(successor, agentIndex + 1, level, alpha, beta)
            if scoreAction[0] > v:
                v = scoreAction[0]
                best = action
                if v > beta:
                    return (v, best)
                alpha = max(v, alpha)
        return (v, best)

    def minValue(self, state, agentIndex, level, alpha, beta):
        v = float("inf")
        best = "Stop"

        if level == self.depth - 1 and agentIndex == (state.getNumAgents() - 1):
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                scoreAction = (self.evaluationFunction(successor), action, alpha, beta)
                if scoreAction[0] < v:
                    v = scoreAction[0]
                    best = action
                    if v < alpha:
                        return (v, best)
                    beta = min(v, beta)
            return (v, best)
        else:
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                scoreAction = self.value(successor, agentIndex + 1, level, alpha, beta)
                if scoreAction[0] < v:
                    v = scoreAction[0]
                    best = action
                    if v < alpha:
                        return (v, best)
                    beta = min(v, beta)
            return (v, best)

    def value(self, state, agentIndex, level, alpha, beta):
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state), "Stop")
        elif agentIndex == state.getNumAgents(): #last ghost
            return self.value(state, 0, level + 1, alpha, beta)
        elif agentIndex == 0: #pacman
            return self.maxValue(state, agentIndex, level, alpha, beta)
        else: #if ghost 
            return self.minValue(state, agentIndex, level, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.value(gameState, 0, 0)[1]

    def maxValue(self, state, agentIndex, level):
        v = float("-inf")
        best = "Stop"
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            scoreAction = self.value(successor, agentIndex + 1, level)
            if scoreAction[0] > v:
                v = scoreAction[0]
                best = action
        return (v, best)

    def expectValue(self, state, agentIndex, level):
        v = 0.0
        best = "Stop"
        moves = state.getLegalActions(agentIndex)
        numMoves = len(moves)

        if level == self.depth - 1 and agentIndex == (state.getNumAgents() - 1):
            for action in moves:
                successor = state.generateSuccessor(agentIndex, action)
                scoreAction = (self.evaluationFunction(successor), action)
                v += float(scoreAction[0]) / float(numMoves)
            return (v, best)
        else:
            for action in moves:
                successor = state.generateSuccessor(agentIndex, action)
                scoreAction = self.value(successor, agentIndex + 1, level)
                v += float(scoreAction[0]) / float(numMoves)
            return (v, best)



    def value(self, state, agentIndex, level):
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state), "Stop")
        elif agentIndex == state.getNumAgents(): #last ghost
            return self.value(state, 0, level + 1)
        elif agentIndex == 0: #pacman
            return self.maxValue(state, agentIndex, level)
        else: #if ghost 
            return self.expectValue(state, agentIndex, level)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Finds the closest power pellet, closest food, and closest scared ghost.
	Weights these along with the number of power pellets left, the number of food left,
	and the number of scared ghosts left.
    """
    
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    powerPellets = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodList = newFood.asList()
    

    food = sys.maxint
    numFood = len(foodList) 
	
    #finding the closest food pellet
    for f in foodList:
	d = util.manhattanDistance(newPos, f)
	food  = min(food, d)
#prevents m from being maxint if there is no food left
    if numFood == 0:
	food = 0
    
    numPellets = len(powerPellets)	
    p = sys.maxint
    #finding the closest power pellet
    for pellet in powerPellets:
	d = util.manhattanDistance(newPos, pellet)
	p = min(d, p)
    
    if numPellets == 0:
	p = 0
    
    #finding the  closest scared ghost
    s = sys.maxint
    numScaredGhosts = 0
    for ghost in newGhostStates:	
	d = util.manhattanDistance(newPos, ghost.getPosition())
	if newScaredTimes[newGhostStates.index(ghost)] > 0:
	     s = min(d,s)
	     numScaredGhosts += 1
	
     #prevents s and g from being maxint if there are no ghosts or there are no scared ghosts or there are no unscared ghosts
    if s == sys.maxint:
	s = 10000


    # Just playing with the weights of the evaluation function until it works.
    #the evalution function to be returned   
    
    r = currentGameState.getScore() + (2.0 / (p + 1)) - (60.0 * numPellets) + (1.0 / (food + 1)) -(2.0 * numFood) + (1.0 / (s+1)) - (30.0 * numScaredGhosts) 

    return r


# Abbreviation
better = betterEvaluationFunction

