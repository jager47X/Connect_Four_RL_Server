package handler;

import request.ParsedRequest;

public class HandlerFactory {
  // routes based on the path. Add your custom handlers here

public static BaseHandler getHandler(ParsedRequest request) {
      switch (request.getPath()) {
          case "/AgentResponse" -> {

              return new AgentResponse();
          }

          default -> {
              return new FallbackHandler();
          }
      }
  }

}
