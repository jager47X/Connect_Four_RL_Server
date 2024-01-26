package request;

public class CustomParser {
    public CustomParser() {
    }

    public static ParsedRequest parse(String request) {
        String[] lines = request.split("(\r\n|\r|\n)");
        String requestLine = lines[0];
        String[] requestParts = requestLine.split(" ");
        ParsedRequest result = new ParsedRequest();
        result.setMethod(requestParts[0]);
        String[] parts = requestParts[1].split("\\?");
        result.setPath(parts[0]);
        String[] pair;
        if (parts.length == 2) {
            System.out.println(parts[1]);
            String[] queryParts = parts[1].split("&");

            for(int i = 0; i < queryParts.length; ++i) {
                pair = queryParts[i].split("=");
                result.setQueryParam(pair[0], pair[1]);
            }
        }

        String body = "";
        boolean emptyLine = false;
        pair = lines;
        int var9 = lines.length;

        for(int var10 = 0; var10 < var9; ++var10) {
            String line = pair[var10];
            if (line.contains(":") && !emptyLine) {
                String[] headerParts = line.split(":");
                String key = headerParts[0].trim();
                String value = headerParts[1].trim();
                result.setHeaderValue(key, value);
                if (key.equalsIgnoreCase("cookie")) {
                    String[] cookieParts = value.trim().split(";");
                    String[] var16 = cookieParts;
                    int var17 = cookieParts.length;

                    for(int var18 = 0; var18 < var17; ++var18) {
                        String cookiePart = var16[var18];
                        String[] cookieStringParts = cookiePart.split("=");
                        result.setCookieValue(cookieStringParts[0], cookieStringParts[1]);
                    }
                }
            }

            if (line.equals("")) {
                emptyLine = true;
            }

            if (emptyLine) {
                body = body + line;
            }
        }

        result.setBody(body);
        return result;
    }
}

