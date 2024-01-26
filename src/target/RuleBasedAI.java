package target;

import java.util.Arrays;
import java.util.Random;

public class RuleBasedAI extends Connect4{
    public RuleBasedAI(){
        super();
    }
    // Check if the given column is a valid move
    private static boolean isValidMove(Board board,int column) {
        return column >= 0 && column < COLS_SIZE &&board.getTile(0,column).getValue() == EMPTY;

    }//end of isValidMove

    // Find the first available row in the given column
    private static int findAvailableRow(Board board, int column) {
        for (int row = ROWS_SIZE- 1; row >= 0; row--) {
            if (board.getTile(row,column).getValue() == EMPTY) {
           //     System.out.println("AI found Available: row:"+",column"+column);
                return row;
            }
        //    System.out.println("AI found Not Available: row:"+row+",column:"+column);
        }
     //   System.out.println("AI did not found Available column at "+ column );
        return -1;  // Column is full
    }// end of findAvailableRow

    // Make a move based on rules (simple rules in this example)
    public static int makeMove(Board board,char activePlayer,char opponent) {
        //pus 1 to return to fit the selction 1-7 instead of 0-6
        // Rule 1: Check if there's a winning move
       // System.out.println("Checking Rule 1");
        int winChoice = isWinningMove(board, activePlayer);
        if (winChoice !=-1){
            return ++winChoice;
        }
        // Rule 2: Block the opponent from winning
       // System.out.println("Checking Rule 2");
        winChoice =isWinningMove(board,opponent);
        if (winChoice !=-1){
            return ++winChoice;
        }
        // Rule 3: Randomly choose an available column
      //  System.out.println("Processing Rule 3");
        Random random = new Random();
        int randomCol;
        do {
            randomCol = random.nextInt(COLS_SIZE);
        } while (!isValidMove(board, randomCol));

     //   System.out.println("Rule 3: Randomly choose an available column");
        return  ++randomCol;
    }


    // Check if the last move is a winning move
    private static int isWinningMove(Board board,   char activePlayer) {
        // Check for horizontal wins
        for (int row = 0; row < ROWS_SIZE; row++) {
            for (int column = 0; column < COLS_SIZE - 3; column++) {
                // Check if within array bounds
                if (board.getTile(row, column).getValue() == EMPTY &&
                        board.getTile(row, column + 1).getValue() == activePlayer &&
                        board.getTile(row, column + 2).getValue() == activePlayer &&
                        board.getTile(row, column + 3).getValue() == activePlayer
                ) {

            //        System.out.println("horizon win FOUND: row:" + row + ",colum:" + column);
                    return column;
                }
                //1.3
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row, column + 1).getValue() == EMPTY &&
                        board.getTile(row, column + 2).getValue() == activePlayer &&
                        board.getTile(row, column + 3).getValue() == activePlayer
                ) {
                    column += 1;
           //         System.out.println("horizon win FOUND: row:" + row + ",colum:" + column);
                    return column;
                }
                //2.2
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row, column + 1).getValue() == activePlayer &&
                        board.getTile(row, column + 2).getValue() == EMPTY &&
                        board.getTile(row, column + 3).getValue() == activePlayer
                ) {
                    column += 2;
          //          System.out.println("horizon win FOUND: row:" + row + ",colum:" + column);
                    return column;
                }
                //3.1
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row, column + 1).getValue() == activePlayer &&
                        board.getTile(row, column + 2).getValue() == activePlayer &&
                        board.getTile(row, column + 3).getValue() == EMPTY
                ) {
                    column += 3;
          //          System.out.println("horizon win FOUND: row:" + row + ",colum:" + column);
                    return column;
                }
            }
        }
  //      System.out.println("horizon win NOT found");




        // Check for vertical wins
        for (int row = 0; row < ROWS_SIZE-3; row++) {
            for (int column = 0; column < COLS_SIZE; column++) {
                // Check if within array bounds
                if (board.getTile(row, column).getValue() == EMPTY &&
                        board.getTile(row + 1, column).getValue() == activePlayer &&
                        board.getTile(row + 2, column).getValue() == activePlayer &&
                        board.getTile(row + 3, column).getValue() == activePlayer
                ) {
       //             System.out.println("vertical win FOUND: row:" + row + ",colum:" + column);
                    return column;
                }
            }

        }
 //       System.out.println("vertical win NOT found");





        // Check for diagonal wins (left to right)
        for (int row = 0; row < ROWS_SIZE-3; row++) {
            for (int column = 0; column <COLS_SIZE - 3; column++) {
                // Check if within array bounds
                if (board.getTile(row, column).getValue() == EMPTY &&
                        board.getTile(row + 1, column + 1).getValue() == activePlayer &&
                        board.getTile(row + 2, column + 2).getValue() == activePlayer &&
                        board.getTile(row + 3, column+3).getValue() == activePlayer
                ) {

          //          System.out.println("diagonal wins (left to right)win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
            }
            for (int column = 0; column <COLS_SIZE - 3; column++) {
                // Check if within array bounds
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row + 1, column + 1).getValue() == EMPTY &&
                        board.getTile(row + 2, column + 2).getValue() == activePlayer &&
                        board.getTile(row + 3, column+3).getValue() == activePlayer
                ) {
                    column += 1;
          //          System.out.println("diagonal wins (left to right)win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
            }
            for (int column = 0; column <COLS_SIZE - 3; column++) {
                // Check if within array bounds
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row + 1, column + 1).getValue() == activePlayer &&
                        board.getTile(row + 2, column + 2).getValue() == EMPTY &&
                        board.getTile(row + 3, column+3).getValue() == activePlayer
                ) {
                    column += 2;
           //         System.out.println("diagonal wins (left to right)win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
            }
            for (int column = 0; column <COLS_SIZE - 3; column++) {
                // Check if within array bounds
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row + 1, column + 1).getValue() == activePlayer &&
                        board.getTile(row + 2, column + 2).getValue() == activePlayer &&
                        board.getTile(row + 3, column+3).getValue() == EMPTY
                ) {
                    column += 3;
            //        System.out.println("diagonal wins (left to right)win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
            }
        }
   //     System.out.println("diagonal wins (left to right)NOT found");
        // Check for diagonal wins (right to left)
        for (int row = 0; row < ROWS_SIZE-4; row++) {
            for (int column = 3; column < COLS_SIZE; column++) {
                // Check if within array bounds
                if (board.getTile(row, column).getValue() ==EMPTY &&
                        board.getTile(row+1, column-1).getValue() == activePlayer &&
                        board.getTile(row+2, column-2 ).getValue() == activePlayer &&
                        board.getTile(row + 3, column-3).getValue() == activePlayer
                ) {

         //           System.out.println("diagonal wins (right to left) win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row+1, column-1).getValue() == EMPTY &&
                        board.getTile(row+2, column-2 ).getValue() == activePlayer &&
                        board.getTile(row + 3, column-3).getValue() == activePlayer
                ) {
                    column-=1;
         //           System.out.println("diagonal wins (right to left) win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row+1, column-1).getValue() == activePlayer &&
                        board.getTile(row+2, column-2 ).getValue() == EMPTY &&
                        board.getTile(row + 3, column-3).getValue() ==  activePlayer
                ) {
                    column-=2;
           //         System.out.println("diagonal wins (right to left) win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
                if (board.getTile(row, column).getValue() == activePlayer &&
                        board.getTile(row+1, column-1).getValue() == activePlayer &&
                        board.getTile(row+2, column-2 ).getValue() == activePlayer &&
                        board.getTile(row + 3, column-3).getValue() == EMPTY
                ) {
                    column-=3;
           //        System.out.println("diagonal wins (right to left) win FOUND: row:" + row + ",colum:" + column);

                    return column;
                }
            }

        }
  //      System.out.println("Win-check NOT found");
        return -1;
    }//end of isWinningMove
}

        /*
    private static boolean isWinningMove(Board board, int row, int col,char activePlayer) {
        // Check horizontally
        int count = 0;
        System.out.println("Check horizontally");
        for (int c = Math.max(0, col - 3); c <= Math.min(COLS_SIZE - 4, col); c++) {
            if (board.getTile(row,c).getValue() ==activePlayer) {
                count++;
                System.out.println("horizon count: "+count+" colum: "+c);
                if (count == 3) {
                    System.out.println("horizon win checked");
                    return true;
                }
            } else {
                count = 0;
            }
        }

        // Check vertically
        System.out.println("Check vertically");
        count = 0;
        for (int r = Math.max(0, row - 3); r <= Math.min(ROWS_SIZE- 4, row); r++) {
            if (board.getTile(r ,col).getValue()  == activePlayer) {
                count++;
                System.out.println("vertical count: "+count+" row: "+r);
                if (count == 3) {
                    System.out.println("vertical win checked");
                    return true;
                }
            } else {
                count = 0;
            }
        }

        // Check diagonally (top-left to bottom-right)
        System.out.println("Check diagonally (top-left to bottom-right)");
        count = 0;
        for (int r = Math.max(0, row - 3), c = Math.max(0, col - 3); r <= Math.min(ROWS_SIZE- 4, row) && c <= Math.min(COLS_SIZE - 4, col); r++, c++) {
            if (board.getTile(r ,c).getValue() == activePlayer) {
                count++;
                System.out.println("diagonal LR count: "+count+" row: "+r);
                System.out.println("vertical count: "+count+" row: "+r);
                if (count == 3) {
                    System.out.println("diagonal LR win checked");
                    return true;
                }
            } else {
                count = 0;
            }
        }

        // Check diagonally (top-right to bottom-left)
        System.out.println("Check diagonally (top-right to bottom-left)");
        count = 0;
        for (int r = Math.max(0, row - 3), c = Math.min(COLS_SIZE - 1, col + 3); r <= Math.min(ROWS_SIZE- 4, row) && c >= Math.max(3, col); r++, c--) {
            if (board.getTile(r,c).getValue() == activePlayer) {
                count++;
                System.out.println("diagonal RL count: "+count+" row: "+r);
                if (count == 3) {
                    System.out.println("diagonal RL win checked");
                    return true;
                }
            } else {
                count = 0;
            }
        }

        return false;
    }

}
*/