package me.utkarshpriyam.Network;

import java.io.*;
import java.util.StringTokenizer;

/**
 * This is the main class which will run the network
 * Eventually this class will be replaced by an
 * interface which will be able to perform actions
 * on the network such as changing the number of layers
 * or changing the number of neurons in a layer.
 *
 * @author Utkarsh Priyam
 * @version 9/4/19
 */
public class Main
{
   /**
    * The default network configuration file paths
    */
   private static String defaultFileConfigPath = "fileConfig.txt";
   private static String defaultNetworkConfigPath = "networkConfig.txt";

   /**
    * This is the public static void main(...) method that
    * will run the network until an interface is created
    * instead.
    *
    * This method takes the classic String[] parameter args.
    *
    * @param args A 1D array of Strings that holds any additional
    *             parameters the method might need in order to run.
    */
   public static void main(String[] args)
   {
      BufferedReader inputStreamReader = new BufferedReader(new InputStreamReader(System.in));

      String inputFilePath, outputsFilePath;
      String weightsFilePath, weightDumpPath, outputDumpPath, otherDumpPath;
      try
      {
         System.out.println("Enter the file path to the file configuration file");
         String fileConfigPath = inputStreamReader.readLine();
         if (fileConfigPath.equals(""))
            fileConfigPath = defaultFileConfigPath;

         File fileConfigFile = new File(fileConfigPath);
         BufferedReader fileConfigReader = new BufferedReader(new FileReader(fileConfigFile));

         // Required
         inputFilePath = fileConfigReader.readLine();
         outputsFilePath = fileConfigReader.readLine();

         // Not Required
         weightsFilePath = fileConfigReader.readLine();
         weightDumpPath = fileConfigReader.readLine();
         outputDumpPath = fileConfigReader.readLine();
         otherDumpPath = fileConfigReader.readLine();

         // Confirmation print
         System.out.println("Using the file config path \"" + fileConfigPath + "\"\n");
      }
      catch (IOException ioException)
      {
         System.out.println("The file configuration file was missing information, badly formatted, or missing");
         ioException.printStackTrace();
         return;
      }

      String networkStructure, runType;
      String lambdaConfig, randomWeightBounds, useRaggedArrays;
      double minimumError;
      int maximumIterationCount;
      try
      {
         System.out.println("Enter the file path to the network configuration file");
         String networkStructureFile = inputStreamReader.readLine();
         if (networkStructureFile.equals(""))
            networkStructureFile = defaultNetworkConfigPath;

         File fileConfigFile = new File(networkStructureFile);
         BufferedReader fileConfigReader = new BufferedReader(new FileReader(fileConfigFile));

         networkStructure = fileConfigReader.readLine();
         runType = fileConfigReader.readLine();

         lambdaConfig = fileConfigReader.readLine();
         minimumError = parseDouble(fileConfigReader.readLine(),0.00001);
         maximumIterationCount = parseInt(fileConfigReader.readLine(),100000);
         randomWeightBounds = fileConfigReader.readLine();
         useRaggedArrays = fileConfigReader.readLine();

         System.out.println("Using the network config path \"" + networkStructureFile + "\"\n");
      }
      catch (IOException ioException)
      {
         System.out.println("The network configuration file was missing information, badly formatted, or missing");
         ioException.printStackTrace();
         return;
      }

      // Read all 3 files
      File weightsFile = new File(weightsFilePath);
      File inputsFile = new File(inputFilePath);
      File outputsFile = new File(outputsFilePath);

      // Create a new Perceptron with the specified dimensions
      String[] networkConfig = networkStructure.split("-");
      int[] networkDimensions = new int[networkConfig.length];
      for (int networkLayerIndex = 0; networkLayerIndex < networkConfig.length; networkLayerIndex++)
         networkDimensions[networkLayerIndex] = parseInt(networkConfig[networkLayerIndex],0);
      boolean useRagged = useRaggedArrays.equals("true");
      Perceptron pdp = new Perceptron(networkDimensions,useRagged);

      // Load lambda configuration information
      StringTokenizer lambdaConfigTokenizer = new StringTokenizer(lambdaConfig);
      double[] lambdaConfigurations = new double[4];
      // Default 0 if missing (in most cases will simply prevent training the network)
      for (int i = 0; i < lambdaConfigurations.length; i++)
         lambdaConfigurations[i] = parseDouble(lambdaConfigTokenizer.nextToken(),0.0);
      pdp.loadLambdaConfig(lambdaConfigurations);

      // Read the weights for the network from the file
      StringTokenizer randomWeightBoundsTokenizer = new StringTokenizer(randomWeightBounds);
      double minWeight = parseDouble(randomWeightBoundsTokenizer.nextToken(),0.0);
      double maxWeight = parseDouble(randomWeightBoundsTokenizer.nextToken(),0.0);
      pdp.readWeights(weightsFile,minWeight,maxWeight);

      // Load in the other stopping condition parameters for the network
      pdp.loadStopConditions(minimumError,maximumIterationCount);

      // Calculated Outputs
      double[][] calculatedOutputs;
      if (runType.equals("run"))
      {
         calculatedOutputs = pdp.runNetwork(inputsFile);
      }
      else if (runType.equals("train"))
      {
         // Train network
         pdp.trainNetwork(inputsFile,outputsFile);

         calculatedOutputs = pdp.runNetwork(inputsFile);
      }
      else if (runType.equals("test"))
      {
         // Nothing done here right now
         calculatedOutputs = new double[0][0];
      }
      else
      {
         System.out.println(runType + " is not a valid run type for this network.");
         System.out.println("The only valid run types are \"run\", \"train\", and \"test\".");
         return;
      }

      try
      {
         double[][][] finalWeights = pdp.weights;
         PrintWriter weightsDumpWriter = new PrintWriter(new BufferedWriter(new FileWriter(weightDumpPath)));
         for (int weightLayerIndex = 0; weightLayerIndex < finalWeights.length; weightLayerIndex++)
         {
            int prevLayerNeuronCount = networkDimensions[weightLayerIndex];
            int nextLayerNeuronCount = networkDimensions[weightLayerIndex + 1];

            for (int prevLayerElementIndex = 0; prevLayerElementIndex < prevLayerNeuronCount; prevLayerElementIndex++)
               for (int nextLayerElementIndex = 0; nextLayerElementIndex < nextLayerNeuronCount; nextLayerElementIndex++)
               {
                  double weightValue = finalWeights[weightLayerIndex][prevLayerElementIndex][nextLayerElementIndex];
                  weightsDumpWriter.print(weightValue + " ");
               }
            weightsDumpWriter.println();
         }
         weightsDumpWriter.close();

         PrintWriter outputsDumpWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputDumpPath)));
         for (double[] testCaseOutputs : calculatedOutputs)
         {
            for (double singularOutput : testCaseOutputs)
               outputsDumpWriter.print(singularOutput + " ");

            outputsDumpWriter.println();
         }
         outputsDumpWriter.close();
      }
      catch (IOException ioException)
      {
         System.out.println("Uncontrolled IOException encountered while logging data to dump files");
         ioException.printStackTrace();
      }
   }

   /**
    * This method parses a integer from a single string token.
    * If the token is not a integer, then it just returns the default value.
    *
    * This method takes two String parameters: nextToken and defaultValue
    *
    * @param nextToken     The token to parse
    * @param defaultValue  The default value to return
    *
    * @return The parsed integer, or the defaultValue if the token cannot be parsed
    */
   private static int parseInt(String nextToken, int defaultValue)
   {
/*
 * The use of 2 return statements in this method is completely
 * intentional as it improves the readability of the method significantly
 * over using a single return and intermediate storage variables.
 */
      try
      {
         return Integer.parseInt(nextToken);
      }
      catch (NumberFormatException ignored)
      {
         return defaultValue;
      }
   }

   /**
    * This method parses a double from a single string token.
    * If the token is not a double, then it just returns the default value.
    *
    * This method takes two String parameters: nextToken and defaultValue
    *
    * @param nextToken     The token to parse
    * @param defaultValue  The default value to return
    *
    * @return The parsed double, or the defaultValue if the token cannot be parsed
    */
   private static double parseDouble(String nextToken, double defaultValue)
   {
/*
 * The use of 2 return statements in this method is completely
 * intentional as it improves the readability of the method significantly
 * over using a single return and intermediate storage variables.
 */
      try
      {
         return Double.parseDouble(nextToken);
      }
      catch (NumberFormatException ignored)
      {
         return defaultValue;
      }
   }
}
