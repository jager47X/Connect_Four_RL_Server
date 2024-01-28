package ReinforceLearning;

import dao.QEntry;
import dao.QTableDao;
import dto.Connect4Dto;
import dto.QTableDto;
import target.Connect4;

import java.util.*;


public abstract class AbstractReinforceLearningAgent2D {

    // Environment-specific variables
    protected Connect4 Environment;
    protected QTableDto QTable;
    protected Connect4Dto connect4Dto;
    static int ROWS;
    static int COLS;
    static int ACTIONS;


    protected AbstractReinforceLearningAgent2D(Connect4Dto connect4Dto, QTableDto ImportedQTable) {
        ROWS=connect4Dto.getGame().getROWS_SIZE();
        COLS=connect4Dto.getGame().getCOLS_SIZE();
        ACTIONS=connect4Dto.getGame().getCOLS_SIZE();
        this.connect4Dto=connect4Dto;
        Environment=connect4Dto.getGame();
        QTable=new QTableDto(ImportedQTable);//process QtableList into one Map
    }

    protected AbstractReinforceLearningAgent2D(Connect4Dto connect4Dto) {
        ROWS=connect4Dto.getGame().getROWS_SIZE();
        COLS=connect4Dto.getGame().getCOLS_SIZE();
        ACTIONS=connect4Dto.getGame().getCOLS_SIZE();
        this.connect4Dto=connect4Dto;
        Environment=connect4Dto.getGame();
        QTable=new QTableDto();
    }



    StringBuilder state = new StringBuilder();
    // Action selection logic


    public int selectAction(Connect4Dto CurrentState) {

            // Exploitation: choose the action with the highest Q-value

            String stateIndex = stateToIndex(CurrentState);

            Set<QEntry> qValues = QTable.getQEntry(stateIndex);

            System.out.println("Q-Values for State " + stateIndex + ":");
            for (QEntry qEntry : qValues) {
                Set<Integer> actions = qEntry.getAction();
                for (Integer action : actions) {
                    System.out.println("QEntry: Action " + action + " reward: " + qEntry.getQValue(action));
                }
            }

            int[] legalActions = getLegalActions(CurrentState);

            return findBestAction(legalActions, qValues);


    }



    public int[] getLegalActions(Connect4Dto CurrentState) {//not =-6

        List<Integer> legalActionsList = new ArrayList<>();
// Iterate over each column to check if it's a legal action
        for (int col = 0; col < COLS; col++) {
            if (CurrentState.getGame().isValidColumn(col)) {
                legalActionsList.add(col+1);
            }
        }

        // Convert the list of legal actions to an array
        int[] legalActions = new int[legalActionsList.size()];
        for (int i = 0; i < legalActions.length; i++) {
            legalActions[i] = legalActionsList.get(i);
        }
        System.out.println("legal action:"+ Arrays.toString(legalActions));
        return legalActions;
    }

    public void updateQValue(String state, int action, double immediateReward, String nextState) {

        Map<Integer, Double> currentQValues = QTable.getQValues(state,action);

        // Ensure that the action is present in the Q-values map
        if (currentQValues.containsKey(action)) {
            double currentQValue = currentQValues.get(action);
            double maxNextQValue = QTable.getMaxQValue(nextState);

            double updatedQValue = (1 - QTable.getLearningRate()) * currentQValue +
                    QTable.getLearningRate() * (immediateReward + QTable.getDiscountFactor() * maxNextQValue);
            QTable.updateQValue(state, action, updatedQValue);
        } else {
            // Handle the case when the action is not present in the Q-values map
            System.out.println("Action " + action + " not present in Q-values for state " + state);
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


}
