package dto;


import target.Board;
import target.Connect4;

import java.util.ArrayList;
import java.util.List;

public abstract class BaseDto {
    protected List<String> HashedData;


    public List<String> getHashedData() {
        return HashedData;
    }

    protected Board board;
    protected Connect4 game;


    public BaseDto(Connect4 game) {
        HashedData=new ArrayList<>(42);
        this.game=game;
        this.board=game.getBoard();
    }
    public BaseDto() {
        HashedData=new ArrayList<>(42);
    }

    public Board getBoard() {
        return game.getBoard();
    }

    public Connect4 getGame() {
        return game;
    }

    public void setGame(Connect4 game) {
        this.game = game;
    }

    public void setBoard(Board board) {
        this.board = board;
    }




    public void addLine() {
        if(HashedData.isEmpty()){
            HashedData.add("P0L0W-1R0");
        }
        HashedData.add(toIndex());
    }

    public String toIndex() {
        StringBuilder state = new StringBuilder( );

        String player;
        if (game.getActivePlayer() == Connect4.PLAYER1){
            player = "1";
        } else {
            player = "2";
        }

        state.append("P");
        state.append(player);

        String location = Integer.toString(game.getLocation(game.getCurrentTurn()));//1-7
        state.append("L");
        state.append(location);

        String winner = Integer.toString(game.getWinner());
        state.append("W");
        state.append(winner);

        String reward;
        if(game.getActivePlayer()==Connect4.PLAYER1){
             reward = Integer.toString(game.getTotalRewardP1());
        }else{
            reward = Integer.toString(game.getTotalRewardP2());
        }
        state.append("R");
        state.append(reward);

        return state.toString();
    }

}

