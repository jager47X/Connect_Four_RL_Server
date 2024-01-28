package response;

import dto.BaseDto;
import dto.Connect4Dto;

import java.util.List;

public class RestApiAppResponse<T extends BaseDto> {

  public final boolean status;
  public final Connect4Dto data;
  public final String message;

  public RestApiAppResponse(boolean status, Connect4Dto data, String message) {
    this.status = status;
    this.data = data;
    this.message = message;
  }
}
