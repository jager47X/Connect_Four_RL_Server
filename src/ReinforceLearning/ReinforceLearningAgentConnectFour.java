package ReinforceLearning;



import dto.Connect4Dto;
import dto.QTableDto;
import target.Connect4;
import target.RuleBasedAI;


public class  ReinforceLearningAgentConnectFour extends AbstractReinforceLearningAgent2D {
    int TotalReward;

    public ReinforceLearningAgentConnectFour (Connect4Dto connect4dto,QTableDto importedQTable) {
        super(connect4dto,importedQTable);
        TotalReward=0;
    }

    public ReinforceLearningAgentConnectFour (Connect4Dto connect4dto) {
        super(connect4dto);
        TotalReward=0;
    }


    public int AResponse(Connect4Dto CurrentState) {
        return selectAction(CurrentState);
    }


}
