package response;

import handler.GsonTool;
import java.util.HashMap;
import java.util.Map;

public class HttpResponseBuilder {

  private Map<String,String> headers = new HashMap<>();
  private String status;
  private String version = "HTTP/1.1";
  private RestApiAppResponse body;
  private String bodyString;

  public HttpResponseBuilder setHeaders(Map<String, String> headers) {
    this.headers = headers;
    return this;
  }

  public HttpResponseBuilder setHeader(String key, String value) {
    headers.put(key, value);
    return this;
  }

  public HttpResponseBuilder setStatus(String status) {
    this.status = status;
    return this;
  }

  public HttpResponseBuilder setVersion(String version) {
    this.version = version;
    return this;
  }

  public HttpResponseBuilder setBody(RestApiAppResponse body) {
    this.body = body;
    return this;
  }

  public HttpResponseBuilder setBody(String body) {
    this.bodyString = body;
    return this;
  }

  public Map<String, String> getHeaders() {
    return headers;
  }

  public String getStatus() {
    return status;
  }

  public String getVersion() {
    return version;
  }

  public RestApiAppResponse getBody() {
    return body;
  }

  public String getBodyString() {
    return bodyString;
  }

  public CustomHttpResponse build(){
    String parsedBody = body != null ? GsonTool.gson.toJson(body) : bodyString;
    return new CustomHttpResponse(headers, status, version, parsedBody);
  }
}
