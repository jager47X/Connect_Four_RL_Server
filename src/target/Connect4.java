package target;

import java.util.ArrayList;
import java.util.List;

public class Connect4{
        protected static final int COLS_SIZE = 7;
        protected static final int ROWS_SIZE = 6;
    public   static final char PLAYER1 = 'X';
    public  static final char PLAYER2 = 'O';
    public static final char EMPTY = '_';




    private static final int SIZE_OF_BOARD=COLS_SIZE*ROWS_SIZE;
        private char activePlayer;
        private char nonActivePlayer;
        private  Board board;
        private int turn;
        private int winner;
        private final int winningPoint=64;
        private int totalRewardP1;
        private int totalRewardP2;
        private final List<Integer> location;
        int totalConnection;
    public Board getBoard() {
        return board;
    }

    public void setBoard(Board board) {
        this.board = board;
    }

    public int getSIZE_OF_BOARD() {
        return SIZE_OF_BOARD;
    }

    public char getNonActivePlayer() {
        return nonActivePlayer;
    }

    public void setNonActivePlayer(char nonActivePlayer) {
        this.nonActivePlayer = nonActivePlayer;
    }

    public Connect4() {
        super();
        this.activePlayer = PLAYER1;
        this.turn=0;
        this.winner=-1;
        this.totalRewardP1 =0;
        this.totalRewardP2 =0;
        this.totalConnection=-1;
        this.board = new Board(COLS_SIZE, ROWS_SIZE,EMPTY,PLAYER1,PLAYER2);
        location=new ArrayList<>(42);
    }

    public int getCurrentTurn() {
        return turn;
    }

    public void setTurn(int turn) {
        this.turn = turn;
    }

    public int getWinner() {
        return winner;
    }

    public void setWinner(int winner) {
        this.winner = winner;
    }

    public char getActivePlayer() {
        return activePlayer;
    }

    public void setActivePlayer(char activePlayer) {
        this.activePlayer = activePlayer;
    }

    public boolean playerDrop(int column_selection) {
        int selection=column_selection;
            column_selection--;//adjust to index of board
            if (isEmpty() && isValidColumn(column_selection)) {

                for (int row = ROWS_SIZE - 1; row >= 0; row--) {

                    if (board.getTile(row,column_selection).getValue() == EMPTY) {
                        setLocation(column_selection);
                  //      System.out.println("selection:"+selection);
                  //      System.out.println("location:"+location+" = col_size:7 *row:"+row+" +col:"+column_selection);
                        board.getTile(row,column_selection).setValue(activePlayer);
                        turn++;
                        calculateReward(getActivePlayer());
                        return true;
                    }
                }
            } else {
        //        System.out.println("(!) Invalid Selection");
            }

            return false;
        }

    public int getCOLS_SIZE() {
        return COLS_SIZE;
    }

    public int getROWS_SIZE() {
        return ROWS_SIZE;
    }

    public boolean isValidColumn(int column) {
            return column >= 0 && column < COLS_SIZE &&board.getTile(0,column).getValue() == EMPTY;
        }

    public int getTotalRewardP1() {
        return totalRewardP1;
    }

    public void setTotalRewardP1(int totalRewardP1) {
        this.totalRewardP1 = totalRewardP1;
    }

    public int getTotalRewardP2() {
        return totalRewardP2;
    }

    public void setTotalRewardP2(int totalRewardP2) {
        this.totalRewardP2 = totalRewardP2;
    }

    public int getLocation(int index) {
        if(location.isEmpty()){
            return 0;
        }
        while(index!=location.size()-1){
            index--;
        }
        return location.get(index);
    }

    public void setLocation(int column) {
        this.location.add(column+1);
    }
    private void calculateReward(char activePlayer) {
        final int baseNumber = 4;
        int maxConnection = 0;

        // Check for horizontal connections
        for (int row = ROWS_SIZE - 1; row >= 0; row--) {
            for (int column = 0; column <= COLS_SIZE - 4; column++) {
                int connection = checkConnection(board, row, column, 0, 1, activePlayer);
                maxConnection = Math.max(maxConnection, connection);
            }
        }

        // Check for vertical connections
        for (int row = ROWS_SIZE - 1; row >= 3; row--) {
            for (int column = 0; column < COLS_SIZE; column++) {
                int connection = checkConnection(board, row, column, -1, 0, activePlayer);
                maxConnection = Math.max(maxConnection, connection);
            }
        }

        // Check for diagonal connections (bottom-left to top-right)
        for (int row = ROWS_SIZE - 1; row >= 3; row--) {
            for (int column = 0; column <= COLS_SIZE - 4; column++) {
                int connection = checkConnection(board, row, column, -1, 1, activePlayer);
                maxConnection = Math.max(maxConnection, connection);
            }
        }

        // Check for diagonal connections (bottom-right to top-left)
        for (int row = ROWS_SIZE - 1; row >= 3; row--) {
            for (int column = 3; column <= COLS_SIZE - 4; column++) {
                int connection = checkConnection(board, row, column, -1, -1, activePlayer);
                maxConnection = Math.max(maxConnection, connection);
            }
        }

        int reward = (int) Math.pow(baseNumber, maxConnection-1);//1 chip=0 2 chip=4

        if (activePlayer == PLAYER1) {
            this.totalRewardP1 = reward;
        } else {
            this.totalRewardP2 = reward;
        }
    }

    private int checkConnection(Board board, int row, int col, int rowDelta, int colDelta, char activePlayer) {
        int connection = 0;

        for (int i = 0; i < 4; i++) {
            int newRow = row + i * rowDelta;
            int newCol = col + i * colDelta;

            // Check if the indices are within the valid range
            if (newRow >= 0 && newRow < ROWS_SIZE && newCol >= 0 && newCol < COLS_SIZE) {
                char value = board.getTile(newRow, newCol).getValue();
                if (value == activePlayer) {
                    connection++;
                } else {
                    break; // Break the loop if no consecutive connection
                }
            } else {
                break; // Break if indices are out of bounds
            }
        }

        return connection;
    }





    public void displayBoard() {
        System.out.println("---------------");
        for (int row = 0; row < ROWS_SIZE; row++) {
            for (int col = 0; col < COLS_SIZE; col++) {
                System.out.print(board.getTile(row,col).getValue() + " ");
            }
            System.out.println();
        }
        System.out.println("---------------");

//
    }

        public void resetBoard() {
          board.initializeBoard();
          this.winner=-1;
          this.totalRewardP1=0;
          this.totalRewardP2=0;
        }


        public boolean winCheck() {

            if (!isEmpty()||turn==42) { // no winner
                winner = 0;
                return true;
            } else if(totalRewardP1==winningPoint){
                winner=1;
                return true;
            } else if (totalRewardP2==winningPoint){
                winner=2;
                return true;
            } else { // game will resume
                winner = -1;
            }
            return false;
        }
        public boolean isEmpty() {//check if any row is empty
            int full = 0;
            for (int row = 0; row < ROWS_SIZE ; row++) {
                if (board.getTile(row,0).getValue()!= EMPTY)
                    full++;
            }
            if (full >= COLS_SIZE) {
                System.out.println("Board is full.");
                return false;
            }
            return true;
        }



}//end of class



