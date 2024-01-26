package target;

public class Tile {

        private int turn;

        private char value; // Represents the state of the tile (e.g., 'X', 'O', '_')

        public Tile(char empty) {
            this.value = empty;
        }
        public Tile() {
        this.value = '_'; // Default value for an empty tile
    }

    public int getTurn() {
        return turn;
    }

    public void setTurn(int turn) {
        this.turn = turn;
    }

    public char getValue() {
            return value;
        }

        public void setValue(char newValue) {
            this.value = newValue;
        }

}
