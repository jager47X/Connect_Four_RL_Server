package dao;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class BaseDao {



    protected static List<String> ImportedData=new ArrayList<>();

    protected static List<List<String>>  ImportedGames=new ArrayList<>();

    public static List<List<String>> getImportedGames() {
        if(ImportedGames==null){
            System.out.println("Warning: ImportedData is Null.");
            return new ArrayList<>();
        }
        return ImportedGames;
    }


    public void import_CSV(String fileName) {
        CSV csv=new CSV(fileName);
        System.out.print("Importing the Data....");
        try (BufferedReader reader = new BufferedReader(new FileReader(csv.getModel()))) {
            String line;
            int gameSet=1;

            while ((line = reader.readLine()) != null) {

                // Process each line as needed
               System.out.println("Processing a CSV line Game:"+gameSet);
                gameSet=processCSVLine(line,gameSet);
            }


            System.out.println("SUCCESS : IMPORTING" + csv.getModel());
        } catch (IOException e) {
            System.err.println("ERROR : " + e.getMessage());
            e.printStackTrace();
        }

    }

    public void exportCSV(List<String>  HashedData,String fileName) {
        CSV csv=new CSV(fileName);
        try (FileWriter writer = new FileWriter(csv.getModel())) {

            for (String hashedLine :  HashedData) {//current game
                writer.append(hashedLine);
                if(hashedLine.contains("-1")){
                    writer.append(",");
                }else{
                    writer.append("\n");
                }
                // Add a newline after each line
            }
            System.out.println("SUCCESS : EXPORTING"+csv.getModel()+".");
        } catch (IOException e) {
            System.err.println("ERROR : " + e.getMessage());
            e.printStackTrace();
        }
    }

    protected int processCSVLine(String line,int gameSet) {
        gameSet++;
        String[] data =line.split(",");//game 1 2 3
            System.out.println("new game");
            ImportedData.clear();

        for (int i = 0; i < data.length; i++) {
            ImportedData.add(i,data[i]);
        }
        List<String> importedGame=new ArrayList<>(ImportedData);
        ImportedGames.add(importedGame);

       System.out.println("data"+Arrays.toString(data));
        return  gameSet;
    }


}
