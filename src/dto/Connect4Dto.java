package dto;

import target.Connect4;

import java.util.List;

public class Connect4Dto extends BaseDto{
    public Connect4Dto(Connect4 game) {
        super(game);
    }
    List<Integer> action;

    public int getLastAction() {
        return action.get(action.size()-1);
    }
    public void addAction(int newAction) {
        action.add(newAction);
    }

}
