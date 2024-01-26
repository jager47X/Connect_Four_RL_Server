package dao;

import java.util.*;

public class QEntry {
    private final Map<Integer, Double> qEntry;

    public QEntry(int newAction, double newQValue) {
        this.qEntry = new HashMap<>();
        this.qEntry.put(newAction, newQValue);
    }
    public QEntry() {
        this.qEntry = new HashMap<>();
    }


    public Set<Integer> getAction() {
        return new HashSet<>(this.qEntry.keySet());
    }
    public QEntry updateQEntry(int newAction, double newQValue, QEntry existingQEntry) {
        Map<Integer, Double> qValues = new HashMap<>();

        // Add/update QValues for all possible actions (assuming actions are integers from 1 to 7)
        for (int i = 1; i <= 7; i++) {
            qValues.put(i, existingQEntry.qEntry.getOrDefault(i, 0.0));
        }

        // Update QValue for the new action
        qValues.put(newAction, newQValue);

        // Clear existing QEntry map and put the updated values
        existingQEntry.qEntry.clear();
        existingQEntry.qEntry.putAll(qValues);

        return existingQEntry;
    }


    public void setQValue(int action, double qValue) {
        if(!qEntry.containsKey(action)){
            qEntry.put(action, qValue);
        }
    }

    public double getQValue(int action) {
        return qEntry.getOrDefault(action, 0.0);
    }

    public Map<Integer, Double> getQEntry() {
        return qEntry;
    }
}
