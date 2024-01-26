package Test;
import ReinforceLearning.ReinforceLearningAgentConnectFour;
import dao.BaseDao;
import dto.Connect4Dto;
import target.Connect4;
import java.util.ArrayList;
import java.util.List;

public class TestAgent {

    public static void main(String[] args) {
        //train agent//choose human or self learn or train agent
        //for human interaction episodes=1;


        List<String> exportingData = new ArrayList<>();
        if(!BaseDao.getImportedGame().isEmpty()){
            for (int i = 0; i < BaseDao.getImportedGame().size(); i++) {
                exportingData.addAll(BaseDao.getImportedGame().get(i));
            }
        }
        for (int i = 0; i <100 ; i++) {
            Connect4 game=new Connect4();
            Connect4Dto connect4Dto=new Connect4Dto(game);
            ReinforceLearningAgentConnectFour Agent = new ReinforceLearningAgentConnectFour(connect4Dto);
            Agent.ReinforceLearning();

        }

        System.out.println("Test:Completed");
    }
}