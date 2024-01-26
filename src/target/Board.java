package target;

public class Board {

    private final Tile[][] board;
    private final char player1_chip ;
    private final char player2_chip ;
    private final char empty_chip;

    private static int column_size;
    private static int row_size;
    public final int length;


    public char getPlayer1_chip() {
        return player1_chip;
    }

    public char getPlayer2_chip() {
        return player2_chip;
    }

    public char getEmpty_chip() {
        return empty_chip;
    }

    public Board(int colsSize, int rowsSize, char empty_chip, char player1_chip, char player2_chip) {
        this.empty_chip=empty_chip;
        this.player1_chip=player1_chip;
        this.player2_chip=player2_chip;
        this.board = new Tile[rowsSize][colsSize];
        column_size=colsSize;
        row_size=rowsSize;
        this.length=colsSize*rowsSize;
        initializeBoard();
    }
    protected void initializeBoard() {
        for (int row = 0; row < row_size; row++) {
            for (int col = 0; col < column_size; col++) {
                board[row][col] = new Tile(empty_chip);
            }
        }
    }

    public Tile getTile(int row, int col) {
        return board[row][col];
    }

    public void setTileValue(int row,int col, char value) {
        board[row][col].setValue(value);
    }

}
