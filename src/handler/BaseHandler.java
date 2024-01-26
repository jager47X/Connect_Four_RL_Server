package handler;

import request.ParsedRequest;
import response.HttpResponseBuilder;

public interface BaseHandler {

  HttpResponseBuilder handleRequest(ParsedRequest request);
}
