package handler;

import request.ParsedRequest;
import response.HttpResponseBuilder;

public class FallbackHandler implements BaseHandler {

  @Override
  public HttpResponseBuilder handleRequest(ParsedRequest request) {
    return new HttpResponseBuilder().setStatus("404 Not Found");
  }
}
