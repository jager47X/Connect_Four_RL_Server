package Test;

import target.Connect4;

import java.util.Scanner;

public class TestConnect4 {


        public static void main(String[] args) {

            int input = 0, turn=0;
            Connect4 game = new Connect4();
            Scanner keyboard = new Scanner(System.in);
            game.setActivePlayer(Connect4.PLAYER2);

            do {
                game.setActivePlayer((game.getActivePlayer() == Connect4.PLAYER2) ? Connect4.PLAYER1 : Connect4.PLAYER2);
                turn++;
                game.setTurn(turn);
                if (game.isEmpty()) {//check if there empty cell
                    do {
                        //displayboard
                        game.displayBoard();
                        System.out.println((game.getActivePlayer() == Connect4.PLAYER1) ? "player1: select between 1-7>> ": "player2: select between 1-7>>" );
                        input = keyboard.nextInt();//human
                        if (game.playerDrop(input)) {
                            System.out.println((game.getActivePlayer() == Connect4.PLAYER1) ? "player1: selected "+ input : "player2: selected" + input);
                            break;
                        }
                    } while (true);
                }// chip is set

                if (game.winCheck()) {//check winner 1 or 2
                    System.out.println((game.getActivePlayer() == Connect4.PLAYER1) ? "player1 Won!" : "player2 Won!");
                    game.displayBoard();
                    System.out.println("total reward player1:"+game.getTotalRewardP1());
                    System.out.println("total reward player2:"+game.getTotalRewardP2());
                    break;
                }else if(game.getWinner()==0){//withdraw
                    System.out.println("Withdraw!");
                    game.displayBoard();
                    System.out.println("total reward player1:"+game.getTotalRewardP1());
                    System.out.println("total reward player2:"+game.getTotalRewardP2());
                    break;
                }
                System.out.println("total reward player1:"+game.getTotalRewardP1());
                System.out.println("total reward player2:"+game.getTotalRewardP2());
            } while (game.isEmpty());

            keyboard.close();

        }




}
