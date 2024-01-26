package ReinforceLearning;



import dto.Connect4Dto;
import dto.QTableDto;
import target.Connect4;
import target.RuleBasedAI;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;


public class  ReinforceLearningAgentConnectFour  extends AbstractReinforceLearningAgent2D {
    int TotalReward;

    public ReinforceLearningAgentConnectFour (Connect4Dto connect4dto, List<QTableDto> qTableDto) {
        super(connect4dto,qTableDto);
        TotalReward=0;
    }
    @Override
    public QTableDto SupervisedLearning() {//use multi-thread //if episodes=1 then vs human if not AI vs AI

        QTableDto qTableDto=new QTableDto(Environment);
        Environment.setActivePlayer(Connect4.PLAYER2);
        Environment.setTurn(0);

        //USE AI
        resetBoard(Environment);
        do {
            if(isGameOver(Environment)){
                Environment.setWinner(0);
                break;
            }
            //update environment/state
            Environment = connect4Dto.getGame();

            if (Environment.getActivePlayer() == Connect4.PLAYER1) {//switch player
                Environment.setActivePlayer(Connect4.PLAYER2);
                Environment.setNonActivePlayer(Connect4.PLAYER1);
            } else {
                Environment.setActivePlayer(Connect4.PLAYER1);
                Environment.setNonActivePlayer(Connect4.PLAYER2);
            }
            while(!Environment.playerDrop(RuleBasedAI.makeMove(Environment.getBoard(),Environment.getActivePlayer(),Environment.getNonActivePlayer()))) {//update move
                Environment.playerDrop(RuleBasedAI.makeMove(Environment.getBoard(),Environment.getActivePlayer(),Environment.getNonActivePlayer()));
            }
            // calculateReward(board.getGame(),);
            if (Environment.winCheck()||Environment.getWinner()==0) {//check winner 1 or 2
                qTableDto.addLine();
                break;
            }else{
                qTableDto.addLine();
            }
        }while(!isGameOver(Environment));


return qTableDto;
    }

    @Override
    public QTableDto ReinforceLearning() {
        QTableDto currentGame = new QTableDto(Environment);
        Environment.setActivePlayer(Connect4.PLAYER1);
        Environment.setTurn(0);

        // USE AI
        resetBoard(Environment);
        do {
            if (isGameOver(Environment)) {
                Environment.setWinner(0);
                break;
            }
            // update environment/state
            System.out.print("Player");
            if (Environment.getActivePlayer() == Connect4.PLAYER1) {// switch player
                Environment.setActivePlayer(Connect4.PLAYER2);
                Environment.setNonActivePlayer(Connect4.PLAYER1);
                System.out.println("1's turn");
            } else {
                Environment.setActivePlayer(Connect4.PLAYER1);
                Environment.setNonActivePlayer(Connect4.PLAYER2);
                System.out.println("2's turn");
            }

            Environment.playerDrop(selectAction());
            StringBuilder state = getState();

            int action = Environment.getLocation(Environment.getCurrentTurn());
            int currentTurn = connect4Dto.getGame().getCurrentTurn();

            Set<Double> setOfRewards = QtableDto.getAllRewards(state.toString(), action);

            if (QtableDto.hasNextState(currentTurn, state.toString())) {
                String nextState = QtableDto.getNextState(connect4Dto.getGame().getCurrentTurn(), state.toString());
                for (Double reward : setOfRewards) {
                    // Update Q-value for each reward in the set
                    updateQValue(state.toString(), action, reward, nextState);
                }
            }

            Environment.displayBoard();
            if (Environment.winCheck() || Environment.getWinner() == 0) {// check winner 1 or 2
                System.out.print("saving into qtableDto:");
                currentGame.addLine();
                break;
            } else {
                System.out.print("saving into qtableDto:");
                currentGame.addLine();
            }
        } while (!isGameOver(Environment));


        return currentGame;
    }

    // Add other necessary methods here...



    //trainAgent RL
    //take Dto as a list
    //after take dto as list reset environment
    //while (back to take dto)
    //export at once

    public void resetBoard(Connect4 game) {
        game.resetBoard();
    }


    private boolean isGameOver(Connect4 game) {
        return game.getWinner() != -1;
    }


    @Override
    public int[] getLegalActions() {//not =-6
        List<Integer> legalActionsList = new ArrayList<>();
// Iterate over each column to check if it's a legal action
            for (int col = 0; col < Environment.getCOLS_SIZE(); col++) {
                if (Environment.isValidColumn(col)) {
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





    @Override
    public int selectAction() {
        return super.selectAction();
    }


}
