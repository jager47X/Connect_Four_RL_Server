package response;

import java.util.Map;
import java.util.Map.Entry;

public class CustomHttpResponse {
  public final Map<String,String> headers;
  public final String status;
  public final String version;
  public final String body;

  public CustomHttpResponse(Map<String, String> headers, String status, String version,
      String body) {
    this.headers = headers;
    this.status = status;
    this.version = version;
    this.body = body;
  }

  public String toString(){
    String headerString = "";
    for(Entry<String,String> header:  headers.entrySet()){
      headerString += header.getKey() + ": " + header.getValue() + "\r\n";
    }
    String res = version + " " + status + "\n" + headerString;

    if(body != null){
      res += "\n" + body;
    }

    return res;
  }
}
