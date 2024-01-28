package Test;

import ReinforceLearning.ReinforceLearningAgentConnectFour;
import com.sun.management.OperatingSystemMXBean;
import dao.BaseDao;
import dao.QTableDao;
import dto.Connect4Dto;
import dto.QTableDto;
import target.Connect4;

import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;



public class TrainAgentRL implements Callable<List<String>> {

    QTableDto Qtable=new QTableDto();
    private final Lock agentLock = new ReentrantLock();
    private static long startTime = System.currentTimeMillis();


    @Override
    public List<String> call() {
        //        System.out.println("Thread " + Thread.currentThread().getId() + " is running.");
        return train();
    }

    private List<String> train() {
        Connect4 game = new Connect4();
        Connect4Dto connect4Dto = new Connect4Dto(game);

        ReinforceLearningAgentConnectFour agent = new ReinforceLearningAgentConnectFour(connect4Dto, Qtable);


        if (agentLock.tryLock()) {
            try {
                Qtable = agent.ReinforceLearning();

                if (Qtable.getHashedData() == null || Qtable.getHashedData().isEmpty()) {
                    //           System.out.println("Thread " + Thread.currentThread().getId() + ": Warning - hashedData is null or empty.");
                    return new ArrayList<>();
                } else {
                    //           System.out.println("Thread " + Thread.currentThread().getId() + ": hashedData is " + Qtable.getHashedData());
                }

                return Qtable.getHashedData();
            } finally {
                agentLock.unlock();
            }
        } else {
            // Handle inability to acquire the lock within the timeout
            System.out.println("Thread " + Thread.currentThread().getId() + ": Unable to acquire agentLock within the timeout.");
            return new ArrayList<>();
        }
    }



    public static void main(String[] args) throws ExecutionException, InterruptedException {
        int thread =2000; // Adjust
        final int nThread=2000;
        final int nTrain=1000000;
        final int nLoop=nTrain/thread;
        long aveProcessTime=0,estimateTotalTime=0;int futureIndex=0;

        QTableDao qTableDao = QTableDao.getInstance();

        List<String> exportingData = new ArrayList<>();
        if(!BaseDao.getImportedGames().isEmpty()){
            for (int i = 0; i < BaseDao.getImportedGames().size(); i++) {
                exportingData.addAll(BaseDao.getImportedGames().get(i));
            }

        }
        int trainIndex=0;
        for (int i = 0; i < nLoop; i++) {
            Thread.sleep(2000);
            // Import game once

            ExecutorService executor = Executors.newFixedThreadPool(nThread);
            List<Future<List<String>>> futures = new ArrayList<>();

            try {
                for (int episode = 0; episode < thread; episode++) {

                    TrainAgentRL trainAgent = new TrainAgentRL();
                    Future<List<String>> future = executor.submit(trainAgent);
                    futures.add(future);
                    System.out.print("Starting Thread: #"+episode+", ");
                    Thread.sleep(monitorCPUUsage());
                }

                // Wait for all threads to finish
                for (Future<List<String>> future : futures) {
                    futureIndex++;
                    trainIndex++;


                    try {

                        System.out.println("Processing thread: " + Thread.currentThread().getId() + ", Index: " + trainIndex);
                        long startTimeThread = System.currentTimeMillis();
                        System.out.print("Saving the game... ");

                        List<String> threadResult = future.get(5, TimeUnit.SECONDS); // Set your timeout value


                        if (threadResult != null) {
                            Thread.sleep(monitorCPUUsage());
                            System.out.print("Thread adding data... ");
                            exportingData.addAll(threadResult);
                            System.out.println(future.state());
                            long endTimeThread = System.currentTimeMillis();
                            System.out.println("Thread execution time: " + (endTimeThread - startTimeThread) + "ms");

                        } else {
                            System.out.println("Thread result is null. Check the TrainAgent.train() method.");
                        }

                        Thread.sleep(monitorCPUUsage());
                        if(trainIndex%10==0){
                            aveProcessTime = System.currentTimeMillis() - startTime;
                            aveProcessTime/=10;
                            startTime = System.currentTimeMillis();
                        }

                        int remainTrain=nTrain-trainIndex;
                        estimateTotalTime = (long) remainTrain * aveProcessTime;
                        Duration totalTime = Duration.ofMillis(estimateTotalTime);
                        System.out.println("Estimate training time: " + formatDuration(totalTime) + " Completed: " + trainIndex + "/" + nTrain);

                    } catch (InterruptedException | ExecutionException | TimeoutException e) {
                        e.printStackTrace();

                        //handle failed trainIndex
                        break;
                        // Handle the TimeoutException as needed
                    }

                }
            } finally {
                executor.shutdownNow(); // Shutdown the executor immediately in case of exceptions
            }

            System.out.print("exporting....");
            qTableDao.exportCSV(exportingData,"Supervised_Learning_policyNetwork.csv");

            Duration totalTime = Duration.ofMillis( System.currentTimeMillis()-startTime);
            System.out.println("Total execution time: " + formatDuration(totalTime));

        }

    }

    private static long monitorCPUUsage() {
        OperatingSystemMXBean osBean = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

        // Print overall CPU usage
        System.out.println("CPU Usage: " +osBean.getCpuLoad() * 100 + "%");
        double wait=10000;

        wait*= osBean.getCpuLoad();
        if(wait<0){
            wait=1000;
        }
        //    System.out.println("wait: " +wait /1000 + "second");

        return (long)wait;
    }
    private static String formatDuration(Duration duration) {
        long minutes = duration.toMinutes();
        long seconds = duration.minusMinutes(minutes).getSeconds();
        return String.format("%d min, %d sec", minutes, seconds);
    }
}
