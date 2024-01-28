package dto;

import dao.BaseDao;
import dao.QEntry;
import dao.QTableDao;
import target.Connect4;


import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class QTableDto extends BaseDto {
    public QTableDto(Connect4 game) {//CSV to dto
        super(game);
    }

    public QTableDto() {// defalut
        this.learningRate = 0.1;
        this.discountFactor = 0.9;
        this.minExplorationRate = 0.1;
        this.explorationDecay = 0.95;
        this.explorationRate=0.9;
        ExportingPolicyNetWork =new HashMap<>();
        ImportedPolicyNetWork =QTableDao.getInstance().getImportedMap();

    }
    public QTableDto(QTableDto imported) {//RL
        this.learningRate = imported.learningRate;
        this.discountFactor = imported.discountFactor;
        this.minExplorationRate = imported.minExplorationRate;
        this.explorationDecay = imported.explorationDecay;
        this.explorationRate= imported.explorationRate;
        ExportingPolicyNetWork =imported.ExportingPolicyNetWork;
        ImportedPolicyNetWork =QTableDao.getInstance().getImportedMap();

    }
    private Map<String, Set<QEntry>> ExportingPolicyNetWork;
    private Map<String, Set<QEntry>> ImportedPolicyNetWork;


    double learningRate;
    double discountFactor;
    double minExplorationRate;
    double explorationDecay;
    double explorationRate;

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getDiscountFactor() {
        return discountFactor;
    }

    public void setDiscountFactor(double discountFactor) {
        this.discountFactor = discountFactor;
    }

    public double getMinExplorationRate() {
        return minExplorationRate;
    }

    public void setMinExplorationRate(double minExplorationRate) {
        this.minExplorationRate = minExplorationRate;
    }

    public double getExplorationDecay() {
        return explorationDecay;
    }

    public void setExplorationDecay(double explorationDecay) {
        this.explorationDecay = explorationDecay;
    }

    public double getExplorationRate() {
        return explorationRate;
    }

    public void setExplorationRate(double explorationRate) {
        this.explorationRate = explorationRate;
    }

    public Set<Double> getAllRewards(String state, int action) {
        if (QTableDao.getInstance().getImportedMap().containsKey(state)) {
            Set<QEntry> qEntrySet = QTableDao.getInstance().getImportedMap().get(state);

            return qEntrySet.stream()
                    .map(qEntry -> qEntry.getQValue(action))
                    .collect(Collectors.toSet());
        }
        return Collections.emptySet(); // or null
    }
    public  void toQtable(List<String> moves) {


        // Create new instances of location and reward lists for each game iteration
        List<Integer> location = new ArrayList<>();
        List<Double> reward = new ArrayList<>();

            for (String move : moves) {//put into List
                location.add(addLocation(move));
                reward.add(addReward(move));
            }

                for (int turn = 0; turn <moves.size()-1; turn++) {//get action and qvalue
                    int action = 0;
                    double qvalue=0.0;
                    if(location.size()>turn){
                        action= location.get(turn+1);
                        qvalue = reward.get(turn+1);
                    }

                    StringBuilder state = new StringBuilder( );
                    for (int index = 0; index < moves.size(); index++) {//add up all action
                        if (index==0) {
                            state.append(0);
                        }else if(index%2==0){
                            state.append(2);
                        }else{
                            state.append(1);
                        }
                        state.append(location.get(index));
                    }

                    int currentTurn=turn+1;
                    System.out.println("turn:"+currentTurn);
                    System.out.println("current Qtable:Saving state:"+state);
                    System.out.println("current Qtable:Saving action:"+action+", reward:"+qvalue);
                    updateQTable(state.toString(),  action, qvalue);
                }
    }

    public  void updateQTable(String state, int action, double qvalue) {
        QEntry qEntry = new QEntry(action, qvalue);

        Set<QEntry> qEntrySet = ImportedPolicyNetWork.computeIfAbsent(state, k -> new HashSet<>());

        // Check if an entry with the same action already exists
        boolean containsAction = qEntrySet.stream()
                .anyMatch(entry -> entry.getAction().contains(action));

        if (containsAction) {
            // Entry with the same action exists, update it if needed
            qEntrySet.forEach(entry -> {
                if (entry.getAction().contains(action)) {
                    entry.setQValue(action, qvalue);
                }
            });
        } else {
            // Entry with the same action doesn't exist, add the new QEntry
            qEntrySet.add(qEntry);
        }
    }

    private  int addLocation(String move) {
        int location=0;
        Pattern pattern = Pattern.compile("P([012])L(\\d+)W(-?\\d+)R(\\d+)");
        Matcher matcher = pattern.matcher(move);
        if (matcher.matches()) {
            location=Integer.parseInt(matcher.group(2));
        }
        return location;
    }
    private  double addReward(String move) {
        double reward=0.0;
        Pattern pattern = Pattern.compile("P([012])L(\\d+)W(-?\\d+)R(\\d+)");
        Matcher matcher = pattern.matcher(move);
        if (matcher.matches()) {
            reward=Double.parseDouble(matcher.group(4));
        }

        return reward;
    }

    public QTableDto converge(QTableDto episode){
        //add the episode to RL Policy Network
        toQtable(episode.getHashedData());

        return this;
    }
    public Map<String, Set<QEntry>> getExportingPolicyNetWork() {
        return ExportingPolicyNetWork;
    }

    public void setExportingPolicyNetWork(Map<String, Set<QEntry>> exportingPolicyNetWork) {
        this.ExportingPolicyNetWork = exportingPolicyNetWork;
    }
    public Set<QEntry> getQEntry(String state) {
        if (!ImportedPolicyNetWork.containsKey(state)) {
            System.out.println("The state is absent on the QTableDao.getInstance().get QTable()");
            // Return a default QEntry indicating the absence of the state
            return Set.of(new QEntry());
        }
        System.out.println("MATCHED FROM CSV:" + state);

        return ImportedPolicyNetWork.get(state);
    }

    public void setQEntry(int action, double newQValue, String state) {
        if (ImportedPolicyNetWork.containsKey(state)) {
            ImportedPolicyNetWork.get(state).add(new QEntry(action, newQValue));
        }
    }


    // Get the maximum Q-value for a given state
    public double getMaxQValue(String state) {
        Set<QEntry> qEntrySet = ImportedPolicyNetWork.getOrDefault(state, Set.of(new QEntry()));

        return qEntrySet.stream()
                .mapToDouble(qEntry -> qEntry.getQValue(1)) // Use any valid actionIndex here
                .max()
                .orElse(0.0);
    }


    // Get the Q-value for a state-action pair
    public Map<Integer, Double> getQValues(String state, int action) {
        Set<QEntry> qEntrySet = ImportedPolicyNetWork.getOrDefault(state, Set.of(new QEntry()));

        return qEntrySet.stream()
                .collect(Collectors.toMap(qEntry -> qEntry.getAction().iterator().next(), qEntry -> qEntry.getQValue(action)));
    }

    public void updateQValue(String state, int action, double updatedQValue) {
        Set<QEntry> qTableEntry = ImportedPolicyNetWork.get(state);

        if (ImportedPolicyNetWork.get(state) != null) {
            // Iterate through the set to find the QEntry with the matching action
            for (QEntry qEntry : qTableEntry) {
                if (qEntry.getQEntry().containsKey(action)) {
                    qEntry.setQValue(action, updatedQValue);
                    //  may need to put the updated QEntry back into the set
                    // qTableEntry.remove(qEntry);
                    // qTableEntry.add(qEntry);
                    ImportedPolicyNetWork.get(state).add(qEntry);
                    break; // Exit the loop once the action is found
                }
            }
        } else {
            System.out.println("State " + state + " not present in Q-table");
        }
    }


    public boolean hasNextState(int currentTurn, String state) {
        int nextTurn = currentTurn + 1;
        return !getNextState(currentTurn, state).equals("ERROR");
    }

    public String getNextState(int currentTurn, String state) {
        int nextTurn = currentTurn + 1;
        for (List<String> importedGame : BaseDao.getImportedGames()) {
            if ((importedGame.get(currentTurn).contains(state))) {
                return importedGame.get(nextTurn);
            }
        }
        return "ERROR";
    }
}
