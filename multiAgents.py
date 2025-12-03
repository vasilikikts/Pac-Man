from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    Common base class for all adversarial agents.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            num_agents = state.getNumAgents()
            legal_actions = state.getAvailableActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (MAX)
                best_score = float('-inf')
                for action in legal_actions:
                    successor = state.generateNextState(agentIndex, action)
                    score = minimax(successor, depth, 1)
                    best_score = max(best_score, score)
                return best_score
            else:  # Ghosts (MIN)
                next_agent = agentIndex + 1
                if next_agent == num_agents:
                    next_agent = 0
                    depth += 1

                best_score = float('inf')
                for action in legal_actions:
                    successor = state.generateNextState(agentIndex, action)
                    score = minimax(successor, depth, next_agent)
                    best_score = min(best_score, score)
                return best_score

        legal_actions = gameState.getAvailableActions(0)
        best_score = float('-inf')
        best_action = None

        for action in legal_actions:
            successor = gameState.generateNextState(0, action)
            score = minimax(successor, 0, 1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            num_agents = state.getNumAgents()
            legal_actions = state.getAvailableActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = float('-inf')
                for action in legal_actions:
                    successor = state.generateNextState(agentIndex, action)
                    value = max(value, alphabeta(successor, depth, 1, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                next_agent = agentIndex + 1
                if next_agent == num_agents:
                    next_agent = 0
                    depth += 1

                value = float('inf')
                for action in legal_actions:
                    successor = state.generateNextState(agentIndex, action)
                    value = min(value, alphabeta(successor, depth, next_agent, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        alpha = float('-inf')
        beta = float('inf')
        best_score = float('-inf')
        best_action = None

        for action in gameState.getAvailableActions(0):
            successor = gameState.generateNextState(0, action)
            score = alphabeta(successor, 0, 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        def expectimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            num_agents = state.getNumAgents()
            legal_actions = state.getAvailableActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (MAX)
                best_score = float('-inf')
                for action in legal_actions:
                    successor = state.generateNextState(agentIndex, action)
                    score = expectimax(successor, depth, 1)
                    best_score = max(best_score, score)
                return best_score
            else:  # Ghosts (EXPECTATION)
                next_agent = agentIndex + 1
                if next_agent == num_agents:
                    next_agent = 0
                    depth += 1

                total = 0
                for action in legal_actions:
                    successor = state.generateNextState(agentIndex, action)
                    total += expectimax(successor, depth, next_agent)
                return total / len(legal_actions)

        best_action = None
        best_score = float('-inf')

        for action in gameState.getAvailableActions(0):
            successor = gameState.generateNextState(0, action)
            score = expectimax(successor, 0, 1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance

    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    #Απόσταση από φαντάσματα
    ghost_penalty = 0
    for ghost in ghosts:
        dist = manhattanDistance(pacman_pos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            if dist < 2:
                ghost_penalty += 200
            elif dist < 4:
                ghost_penalty += 100
        else:
            #Αν είναι τρομαγμένο, προσπάθησε να το πλησιάσεις
            ghost_penalty -= 50 / (dist + 1)

    #Απόσταση από φαγητό
    food_score = 0
    if food:
        min_food_dist = min(manhattanDistance(pacman_pos, f) for f in food)
        food_score = 10.0 / (min_food_dist + 1)
    total_food_penalty = len(food) * 5  #όσο περισσότερα, τόσο χειρότερα

    #Κάψουλες
    capsule_bonus = 0
    if capsules:
        min_capsule_dist = min(manhattanDistance(pacman_pos, c) for c in capsules)
        capsule_bonus = 10.0 / (min_capsule_dist + 1)
    total_capsule_penalty = len(capsules) * 20

    #Τελικό σκορ
    final_score = (
            score
            + food_score
            + capsule_bonus
            - ghost_penalty
            - total_food_penalty
            - total_capsule_penalty
    )

    return final_score


# Abbreviation
better = betterEvaluationFunction
