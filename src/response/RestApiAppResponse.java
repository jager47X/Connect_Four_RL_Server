package response;

import dto.BaseDto;
import java.util.List;

public class RestApiAppResponse<T extends BaseDto> {

  public final boolean status;
  public final List<T> data;
  public final String message;

  public RestApiAppResponse(boolean status, List<T> data, String message) {
    this.status = status;
    this.data = data;
    this.message = message;
  }
}
