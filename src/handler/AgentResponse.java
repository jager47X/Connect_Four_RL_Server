package handler;

import ReinforceLearning.ReinforceLearningAgentConnectFour;
import dto.Connect4Dto;
import dto.QTableDto;
import request.ParsedRequest;
import response.HttpResponseBuilder;
import response.RestApiAppResponse;

public class AgentResponse implements BaseHandler {

    public HttpResponseBuilder handleRequest(ParsedRequest request) {
        Connect4Dto connect4Dto = (Connect4Dto)GsonTool.gson.fromJson(request.getBody(), Connect4Dto.class);

        connect4Dto.getGame().playerDrop(connect4Dto.getLastAction());
        ReinforceLearningAgentConnectFour agent=new ReinforceLearningAgentConnectFour(connect4Dto);
        int AIResponse= agent.AResponse(connect4Dto);
        connect4Dto.addAction(AIResponse);
        if(AIResponse>=1&&AIResponse<=7){
            RestApiAppResponse<QTableDto> res = new RestApiAppResponse<>(true, connect4Dto, "success");
            return (new HttpResponseBuilder()).setStatus("200 OK").setBody(res);
        }
        RestApiAppResponse<QTableDto> res = new RestApiAppResponse<>(true, connect4Dto, "fail");
        return (new HttpResponseBuilder()).setStatus("200 OK").setBody(res);
    }
}
//keep adding action to list of action int the connect4Dto