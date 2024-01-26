package dto;

import dao.BaseDao;
import dao.QEntry;
import dao.QTableDao;
import target.Connect4;


import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class QTableDto extends BaseDto {
    public QTableDto(Connect4 game) {//CSV to dto
        super(game);
    }

    public QTableDto() {//CSV to dto

    }

    double learningRate = 0.1;
    double discountFactor = 0.9;
    double minExplorationRate = 0.1;
    double explorationDecay = 0.95;
    double explorationRate=1-learningRate;


    public double getExplorationRate() {
        return explorationRate;
    }

    public void setExplorationRate(double explorationRate) {
        this.explorationRate = explorationRate;
    }

    public Set<Double> getAllRewards(String state, int action) {
        if (QTableDao.getInstance().getQTable().containsKey(state)) {
            Set<QEntry> qEntrySet = QTableDao.getInstance().getQTable().get(state);

            return qEntrySet.stream()
                    .map(qEntry -> qEntry.getQValue(action))
                    .collect(Collectors.toSet());
        }
        return Collections.emptySet(); // or null
    }

    public Set<QEntry> getQEntry(String state) {
        if (!QTableDao.getInstance().getQTable().containsKey(state)) {
            System.out.println("The state is absent on the QTableDao.getInstance().get QTable()");
            // Return a default QEntry indicating the absence of the state
            return Set.of(new QEntry());
        }
        System.out.println("MATCHED FROM CSV:" + state);

        return QTableDao.getInstance().getQTable().get(state);
    }

    public void setQEntry(int action, double newQValue, String state) {
        if (QTableDao.getInstance().getQTable().containsKey(state)) {
            QTableDao.getInstance().getQTable().get(state).add(new QEntry(action, newQValue));
        }
    }


    // Get the maximum Q-value for a given state
    public double getMaxQValue(String state) {
        Set<QEntry> qEntrySet = QTableDao.getInstance().getQTable().getOrDefault(state, Set.of(new QEntry()));

        return qEntrySet.stream()
                .mapToDouble(qEntry -> qEntry.getQValue(1)) // Use any valid actionIndex here
                .max()
                .orElse(0.0);
    }


    // Get the Q-value for a state-action pair
    public Map<Integer, Double> getQValues(String state, int action) {
        Set<QEntry> qEntrySet = QTableDao.getInstance().getQTable().getOrDefault(state, Set.of(new QEntry()));

        return qEntrySet.stream()
                .collect(Collectors.toMap(qEntry -> qEntry.getAction().iterator().next(), qEntry -> qEntry.getQValue(action)));
    }

    public void updateQValue(String state, int action, double updatedQValue) {
        Set<QEntry> qTableEntry = QTableDao.getInstance().getQTable().get(state);

        if (qTableEntry != null) {
            // Iterate through the set to find the QEntry with the matching action
            for (QEntry qEntry : qTableEntry) {
                if (qEntry.getQEntry().containsKey(action)) {
                    qEntry.setQValue(action, updatedQValue);
                    //  may need to put the updated QEntry back into the set
                    // qTableEntry.remove(qEntry);
                    // qTableEntry.add(qEntry);
                    break; // Exit the loop once the action is found
                }
            }
        } else {
            System.out.println("State " + state + " not present in Q-table");
        }
    }


    public boolean hasNextState(int currentTurn, String state) {
        int nextTurn = currentTurn + 1;

        for (List<String> importedGame : BaseDao.getImportedGame()) {
            if (importedGame.size() > currentTurn && importedGame.get(currentTurn).contains(state) && (importedGame.size() > nextTurn)) {
                return true;
            }
        }
        return false;
    }

    public String getNextState(int currentTurn, String state) {
        int nextTurn = currentTurn + 1;
        for (List<String> importedGame : BaseDao.getImportedGame()) {
            if ((importedGame.get(currentTurn).contains(state))) {
                return importedGame.get(nextTurn);
            }
        }
        return "ERROR";
    }
}
