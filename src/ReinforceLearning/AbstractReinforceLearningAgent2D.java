package ReinforceLearning;

import dao.QEntry;
import dto.Connect4Dto;
import dto.QTableDto;
import target.Connect4;

import java.util.*;

interface ReinforceLearningAgent2D{
    QTableDto SupervisedLearning();
    QTableDto ReinforceLearning();
    int[] getLegalActions();
    int selectAction();
}
public abstract class AbstractReinforceLearningAgent2D implements ReinforceLearningAgent2D{

    // Environment-specific variables
    protected Connect4 Environment;
    protected List<QTableDto> QtableList;
    protected Connect4Dto connect4Dto;
    static int ROWS;
    static int COLS;

    static int ACTIONS;


    public Connect4 getEnvironment() {
        return Environment;
    }

    public void setEnvironment(Connect4 environment) {
        Environment = environment;
    }

    protected AbstractReinforceLearningAgent2D(Connect4Dto connect4Dto,List<QTableDto> QtableList) {
        ROWS=connect4Dto.getGame().getROWS_SIZE();
        COLS=connect4Dto.getGame().getCOLS_SIZE();
        ACTIONS=connect4Dto.getGame().getCOLS_SIZE();
        this.connect4Dto=connect4Dto;


        this.QtableList.addAll(QtableList);
        QTableDto qTableDto=this.QtableList.get(QtableList.size()-1);
        this.explorationRate= qTableDto.getExplorationRate();
        this.learningRate=1-this.explorationRate;
        connect4Dto.getGame().resetBoard();
        Environment=connect4Dto.getGame();
    }


    // Hyper-parameters
    double learningRate;
    final double discountFactor=0.8;
    double explorationRate;
    final double minExplorationRate=0.1;
    final  double explorationDecay=0.95;


    StringBuilder state = new StringBuilder();
    // Action selection logic

    @Override
    public int selectAction() {
        if (Math.random() < explorationRate) {
            // Exploration: choose a random action
            System.out.println("Exploration");
            int[] legalActions = getLegalActions();
            if (explorationRate > minExplorationRate) {
                explorationRate = explorationRate * explorationDecay;
            }
            return legalActions[new Random().nextInt(legalActions.length)];
        } else {
            // Exploitation: choose the action with the highest Q-value

            String stateIndex = stateToIndex(this.connect4Dto);

            Set<QEntry>qValues=new HashSet<>();
            List<Set<QEntry>>QValuesList =new ArrayList<>();
            for (int i = 0; i < QtableList.size()-1; i++) {
                qValues.addAll(QtableList.get(i).getQEntry(stateIndex));
                QValuesList.add(qValues);
                qValues.clear();
            }


            System.out.println("Q-Values for State " + stateIndex + ":");
            for (QEntry qEntry : qValues) {
                Set<Integer> actions = qEntry.getAction();
                for (Integer action : actions) {
                    System.out.println("QEntry: Action " + action + " reward: " + qEntry.getQValue(action));
                }
            }

            int[] legalActions = getLegalActions();

            return findBestAction(legalActions, qValues);

        }
    }




    private int findBestAction(int[] legalActions, Set<QEntry> qValues) {
        double bestQValue = 0.0; // initialize with the minimum possible value
        List<Integer> bestActions = new ArrayList<>();

        for (QEntry qEntry : qValues) {
            Set<Integer> actions = qEntry.getAction();
            for (Integer action : actions) {
                double qValue = qEntry.getQValue(action);
                if (qValue > bestQValue) {
                    bestQValue = qValue;
                    bestActions.clear();
                    bestActions.add(action);
                } else if (qValue == bestQValue) {
                    bestActions.add(action);
                }
            }
        }

        if (!bestActions.isEmpty()) {
            int randomIndex = new Random().nextInt(bestActions.size());
            int bestAction = bestActions.get(randomIndex);

            System.out.println("Returning the best action: " + bestAction);
            return bestAction;
        }

        System.out.println("BestQValue is " + bestQValue + " returning random choice");
        return legalActions[new Random().nextInt(legalActions.length)];
    }



    public void updateQValue(String state, int action, double immediateReward, String nextState) {
        Map<Integer, Double> currentQValues = QtableDto.getQValues(state,action);

        // Ensure that the action is present in the Q-values map
        if (currentQValues.containsKey(action)) {
            double currentQValue = currentQValues.get(action);
            double maxNextQValue = QtableDto.getMaxQValue(nextState);

            double updatedQValue = (1 - learningRate) * currentQValue +
                    learningRate * (immediateReward + discountFactor * maxNextQValue);
            QtableDto.updateQValue(state, action, updatedQValue);
        } else {
            // Handle the case when the action is not present in the Q-values map
            System.out.println("Action " + action + " not present in Q-values for state " + state);
        }
    }


    private String stateToIndex(Connect4Dto dto) {
        Connect4 currentBoard = dto.getGame();
        int turn = currentBoard.getCurrentTurn();

        System.out.println("currentTurn " + turn);

        if (turn == 0) {
            state=new StringBuilder();
            state.append("00");
        }else{
            if (currentBoard.getActivePlayer()==Connect4.PLAYER1) {
                state.append("1");
            } else {
                state.append("2");
            }
            state.append(currentBoard.getLocation(turn));
        }


        System.out.println("Current state: " + state);
        return state.toString();
    }

    public StringBuilder getState() {
        return state;
    }

    public void setState(StringBuilder state) {
        this.state = state;
    }
}
