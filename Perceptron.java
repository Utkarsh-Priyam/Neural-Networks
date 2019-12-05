package me.utkarshpriyam.Network;

import java.io.*;
import java.util.*;

/**
 * This is the Perceptron class.
 * It represents a perceptron and uses
 * pdp (parallel distributive processing).
 *
 * This class can be instantiated either
 * by an array containing all layer sizes,
 * or by 2 integers for the input and output layer sizes along
 * with an array to that gives the sizes of the hidden layers.
 *
 * The pdp network reads its weight values and its inputs from
 * files named inputs.txt and weights.txt.
 *
 * The pdp network can run on three different modes:
 * Running, Training, and Testing
 *
 * In Running Mode (this is for actually using the network):
 *   - The network will output the final raw information it calculates
 *   - The network will not take any predicted output values
 *      - As a result, the network will neither train its weights
 *        not return an error value
 *
 * In Training Mode (as the name suggests, to train the network):
 *   - The network will take in both the inputs and the expected outputs
 *   - The network will automatically calculate its error and
 *     use the method of gradient descent to adjust its weights
 *     in order to reduce that aforementioned error
 *   - This uses fully generalized back-propagation!!
 *
 * TO BE IMPLEMENTED:
 * In Testing Mode (this is a blend of the previous two modes):
 *   - The network will take both the inputs and the expected outputs.
 *     However, as this is not a training exercise, the network will not
 *     update its weights in order to minimize error. It will simply calculate
 *     and return the error
 *
 * @author Utkarsh Priyam
 * @version 9/4/19
 */
public class Perceptron
{
   /**
    * This boolean constant dictates whether to instantiate the
    * neuron and edge arrays as a full block (which has wasted space)
    * or as a ragged array (no wasted space)
    */
   private final boolean GENERATE_RAGGED_ARRAYS;

   /**
    * This double value is the learning factor for this pdp network.
    * It is the step size that dictates how "fast" or "slowly" the
    * network adjusts its weights (whether it takes big or small
    * steps in the "downhill direction" (method of gradient descent).
    */
   double lambda;

   /**
    * The private double lambdaChange is a configurable double that tells
    * how much to multiply or divide the lambda learning factor by while
    * training the network.
    *
    * The lambdaMinCap and lambdaMaxCap are two configurable doubles which
    * bound lambda do the range (lambdaMinCap,lambdaMaxCap].
    *
    * If lambda hits the maximum cap, it is simply bounded to that value.
    * On the other hand, if lambda hits the minimum cap, then the training
    * procedure terminates and the program tells that it ended due to the
    * lambda going below a minimum threshold (lambdaMinCap).
    */
   private double lambdaChange;
   private double lambdaMinCap, lambdaMaxCap;

   /**
    * This package-private method takes in a double array that holds
    * all of the hyper-parameters relating to the lambda learning factor.
    *
    * The array must be at least of length 4
    * (or an ArrayIndexOutOfBound exception is thrown)
    *
    * The array's 4 values must be, in order,
    * {lambda, lambdaChange, lambdaMinCap, lambdaMaxCap}.
    *
    * Of course, lambdaMinCap < lambda < lambdaMaxCap is a requirement.
    *
    * @param lambdaConfig  The double array that holds all the configurable
    *                      hyper-parameters for the lambda learning factor.
    */
   void loadLambdaConfig(double[] lambdaConfig)
   {
      lambda = lambdaConfig[0];
      lambdaChange = lambdaConfig[1];
      lambdaMinCap = lambdaConfig[2];

      if (lambdaConfig[3] < 0.0)
         lambdaMaxCap = Double.MAX_VALUE;
      else
         lambdaMaxCap = lambdaConfig[3];
   }

   /**
    * These two private variables hold the minimum error that needs to be
    * achieved in order to declare success while training and the maximum
    * number of iterations before the network will stop training.
    */
   private double minimumError;
   private int maximumIterationCount;

   /**
    * There two values are instance variables that hold information regarding the termination
    * conditions of the network training. These variable values only hold meaning if the
    * network was actually training. Otherwise, they hold the default values (0 and 0.0).
    */
   int iterationCounter;
   double maximumTestCaseError;

   /**
    * This package-private method takes in a double telling the minimum error that needs to be
    * achieved in order to declare success while training. It also takes in an integer which
    * gives a maximum number of iterations before the network will stop training.
    *
    * @param minError   Minimum error to succeed
    * @param maxItCount Maximum number of iterations to complete before terminating
    */
   void loadStopConditions(double minError, int maxItCount)
   {
      minimumError = minError;
      maximumIterationCount = maxItCount;
   }

   /**
    * The layerCounts array stores the number of neurons
    * in each layer of the network.
    *
    * The activations array stores the activation values
    * of those neurons (stored as layer.neuron)
    *
    * The weights array stores the weights of the edges
    * of the network in a 3D array
    * (stored as layer.prevNeuron.nextNeuron)
    */
   private int[] layerCounts;
   private double[][] activations, unboundedActivations;
   double[][][] weights;

   /**
    * These four ints give the lengths and indices for a couple of
    * special locations within the underlying arrays of the network.
    *
    * numNeuronLayers:  The number of neuron layers in the network
    * numWeightLayers:  The number of weight layers in the network
    * inputLayer:       The index of the input layer
    * outputLayer:      The index of the output layer
    */
   private int numNeuronLayers, numWeightLayers, inputLayer, outputLayer;

   /**
    * These two double arrays store values necessary for generalized
    * back-propagation. The deltaWeights array holds the delta weights
    * used during gradient descent, and the omega array stores the omega
    * values used during back-propagation (as per the design document).
    */
   private double[][][] deltaWeights;
   private double[][] omega;

   /**
    * This is the first constructor for the Perceptron class.
    * It takes in a single array that contains the counts
    * of the number of neurons in each layer of the network.
    *
    * @param layerCounts      The array which contains the number of neurons in each layer of the network
    * @param useRaggedArrays  A boolean flag that determines whether or not the underlying arrays
    *                         of the network are ragged arrays or boxed arrays
    *
    * @throws IllegalArgumentException This method throws an IllegalArgumentException
    *                                  if the parameter passed does not have at least
    *                                  2 values (for the input and output layers).
    */
   Perceptron(int[] layerCounts, boolean useRaggedArrays)
   {
      // Set the indices for the most important layers
      numNeuronLayers = layerCounts.length;
      numWeightLayers = numNeuronLayers - 1;
      inputLayer = 0;
      outputLayer = numNeuronLayers - 1;

      // Set whether or not to use ragged arrays
      GENERATE_RAGGED_ARRAYS = useRaggedArrays;

      // Throw an IllegalArgumentException if not enough layers (at least 3) are passed
      if (numNeuronLayers < 3)
         throw new IllegalArgumentException("Not enough layers in network");

      // Adjust the layer counts to ensure that every layer has at least 1 neuron
      setMinimumAllowed(layerCounts,1);

      // Store the layers array
      this.layerCounts = layerCounts;
      // Shorten the layer counts array to exclude the input and output counts
      int[] innerLayerCounts = new int[numNeuronLayers - 2];
      for (int innerLayerIndex = 0; innerLayerIndex < innerLayerCounts.length; innerLayerIndex++)
         innerLayerCounts[innerLayerIndex] = layerCounts[innerLayerIndex + 1];

      // Generate the neuron and edge arrays
      generateNeuronsAndEdgesArrays(layerCounts[inputLayer], innerLayerCounts, layerCounts[outputLayer]);
   }

   /**
    * This is the second constructor for the Perceptron class.
    * It takes in 2 integers which tell the number of neurons
    * in the input and output layers of the network.
    * It also takes in a single array that contains the counts
    * of the number of neurons in each hidden layer of the network.
    *
    * @param numInputs         The number of neurons in the input layer of the network
    * @param hiddenLayersCount The array which contains the number of
    *                          neurons in each hidden layer of the network
    * @param numOutputs        The number of neurons in the output layer of the network
    * @param useRaggedArrays   A boolean flag that determines whether or not the underlying arrays
    *                          of the network are ragged arrays or boxed arrays
    */
   Perceptron(int numInputs, int[] hiddenLayersCount, int numOutputs, boolean useRaggedArrays)
   {
      // Set the indices for the most important layers
      numNeuronLayers = hiddenLayersCount.length + 2;
      numWeightLayers = numNeuronLayers - 1;
      inputLayer = 0;
      outputLayer = numNeuronLayers - 1;

      // Set whether or not to use ragged arrays
      GENERATE_RAGGED_ARRAYS = useRaggedArrays;

      // Adjust the layer counts to ensure that every layer has at least 1 neuron
      if (numInputs < 1)
         numInputs = 1;
      if (numOutputs < 1)
         numOutputs = 1;
      setMinimumAllowed(hiddenLayersCount,1);

      // Compact all the layers data into one single array
      layerCounts = new int[numNeuronLayers];
      layerCounts[inputLayer] = numInputs;
      System.arraycopy(hiddenLayersCount, 0, layerCounts, 1, hiddenLayersCount.length);
      layerCounts[outputLayer] = numOutputs;

      // Generate the neuron and edge arrays
      generateNeuronsAndEdgesArrays(numInputs, hiddenLayersCount, numOutputs);
   }

   /**
    * Goes through the given array and ensures that all values of the array
    * are at least as large as the given input integer
    * @param hiddenLayersCount The array to read and modify
    * @param minimumValue      The minimum allowed value
    */
   private void setMinimumAllowed(int[] hiddenLayersCount, int minimumValue)
   {
      for (int i = 0; i < hiddenLayersCount.length; i++)
         hiddenLayersCount[i] = Math.max(hiddenLayersCount[i], minimumValue);
   }

   /**
    * This method handles the generation of the
    * neuron and edges arrays for this pdp network.
    *
    * It takes in the same three parameters
    * as the second constructor (which takes 3 parameters).
    *
    * It takes in 2 integers which tell the number of neurons
    * in the input and output layers of the network.
    * It also takes in a single array that contains the counts
    * of the number of neurons in each hidden layer of the network.
    *
    * @param numInputs         The number of neurons in the input layer of the network
    * @param hiddenLayersCount The array which contains the number of
    *                          neurons in each hidden layer of the network
    * @param numOutputs        The number of neurons in the output layer of the network
    */
   private void generateNeuronsAndEdgesArrays(int numInputs, int[] hiddenLayersCount, int numOutputs)
   {
      // Generate the underlying arrays of the network
      if (GENERATE_RAGGED_ARRAYS)
         // Generate Ragged Arrays
         generateArraysRagged(numInputs,hiddenLayersCount,numOutputs);
      else
         // Generate the "normal" full block arrays
         generateArraysRegular(numInputs,hiddenLayersCount,numOutputs);
   }

   /**
    * This method is called by generateNeuronsAndEdgesArrays(...)
    * if the boolean constant GENERATE_RAGGED_ARRAYS is true.
    * In this case, the neuron and edges arrays for this
    * pdp network are generated as ragged arrays.
    *
    * This method takes the exact same parameters as the
    * generateNeuronsAndEdgesArrays(...) method.
    *
    * It takes in 2 integers which tell the number of neurons
    * in the input and output layers of the network.
    * It also takes in a single array that contains the counts
    * of the number of neurons in each hidden layer of the network.
    *
    * @param numInputs         The number of neurons in the input layer of the network
    * @param hiddenLayersCount The array which contains the number of
    *                          neurons in each hidden layer of the network
    * @param numOutputs        The number of neurons in the output layer of the network
    */
   private void generateArraysRagged(int numInputs, int[] hiddenLayersCount, int numOutputs)
   {
      // Generate Neurons Array - Total Layers = Input + Hidden + Output = 2 + numHidden
      activations = new double[numNeuronLayers][];
      unboundedActivations = new double[numNeuronLayers][];
      omega = new double[numNeuronLayers][];

      // Set Input Neurons Array length
      activations[0] = new double[numInputs];
      unboundedActivations[0] = new double[numInputs];
      omega[0] = new double[numInputs];
      // Set Hidden Neurons Array length
      for (int i = 1; i < numNeuronLayers - 1; i++)
      {
         activations[i] = new double[hiddenLayersCount[i-1]];
         unboundedActivations[i] = new double[hiddenLayersCount[i-1]];
         omega[i] = new double[hiddenLayersCount[i-1]];
      }
      // Set Output Neurons Array length
      activations[outputLayer] = new double[numOutputs];
      unboundedActivations[outputLayer] = new double[numOutputs];
      omega[outputLayer] = new double[numOutputs];

      // Generate Weights Arrays - Total Layers = numNeurons - 1
      weights = new double[numWeightLayers][][];
      deltaWeights = new double[numWeightLayers][][];

      // Generate Second and Third Array Dimensions: Length = numNeurons in prev layer, next layer
      weights[0] = new double[numInputs][hiddenLayersCount[0]];
      deltaWeights[0] = new double[numInputs][hiddenLayersCount[0]];
      for (int m = 1; m < numWeightLayers - 1; m++)
      {
         weights[m] = new double[hiddenLayersCount[m - 1]][hiddenLayersCount[m]];
         deltaWeights[m] = new double[hiddenLayersCount[m - 1]][hiddenLayersCount[m]];
      }
      weights[numWeightLayers - 1] = new double[hiddenLayersCount[numWeightLayers - 2]][numOutputs];
      deltaWeights[numWeightLayers - 1] = new double[hiddenLayersCount[numWeightLayers - 2]][numOutputs];
   }

   /**
    * This method is called by generateNeuronsAndEdgesArrays(...)
    * if the boolean constant GENERATE_RAGGED_ARRAYS is false.
    * In this case, the neuron and edges arrays for this
    * pdp network are not generated as ragged arrays.
    * They are instead generated as full 2D/3D blocks.
    *
    * This method takes the exact same parameters as the
    * generateNeuronsAndEdgesArrays(...) method.
    *
    * It takes in 2 integers which tell the number of neurons
    * in the input and output layers of the network.
    * It also takes in a single array that contains the counts
    * of the number of neurons in each hidden layer of the network.
    *
    * @param numInputs         The number of neurons in the input layer of the network
    * @param hiddenLayersCount The array which contains the number of
    *                          neurons in each hidden layer of the network
    * @param numOutputs        The number of neurons in the output layer of the network
    */
   private void generateArraysRegular(int numInputs, int[] hiddenLayersCount, int numOutputs)
   {
      // Count number maximum number of neurons in network
      // The number will serve as the 2nd dimension for the neurons array
      // and as the 2nd and 3rd dimensions for the weights array
      int maxNumNeurons = Math.max(numInputs,numOutputs);
      for (int numNeurons: hiddenLayersCount)
         if (maxNumNeurons < numNeurons)
            maxNumNeurons = numNeurons;

      // Generate all arrays
      activations = new double[numNeuronLayers][maxNumNeurons];
      unboundedActivations = new double[numNeuronLayers][maxNumNeurons];
      omega = new double[numNeuronLayers][maxNumNeurons];

      weights = new double[numWeightLayers][maxNumNeurons][maxNumNeurons];
      deltaWeights = new double[numWeightLayers][maxNumNeurons][maxNumNeurons];
   }

   /**
    * This method can be called by the network handler class in order to
    * have the pdp network read the weight values stored in the given file.
    *
    * This method takes the single parameter value weightsFile,
    * which has all of the weights for the network stored
    * in a specific ordering and organization within the file.
    *
    * @param weightsFile       The file which holds all of the weights for the network
    * @param minRandomWeight   The minimum value for weight randomization
    * @param maxRandomWeight   The maximum value for weight randomization
    *
    * @throws RuntimeException This method throws a runtime exception if anything
    *                          goes wrong during the file-reading process.
    *                          This method also prints out the stack trace
    *                          of the original error.
    */
   void readWeights(File weightsFile, double minRandomWeight, double maxRandomWeight)
   {
      try
      {
         // BufferedReader w can read all the weights out of the weightsFile file
         BufferedReader w = new BufferedReader(new FileReader(weightsFile));

         // Iterate through all the different weights layers
         for (int m = 0; m < numNeuronLayers - 1; m++)
         {
            // Make sure the next line is not null
            String textLine = w.readLine();
            if (textLine == null)
               textLine = "";

            // Make a StringTokenizer to read the line
            StringTokenizer weightsLine = new StringTokenizer(textLine);

            // Now iterate over all the edges in the layer with index m

            // Iterate over the neurons in layer m first
            for (int jk = 0; jk < layerCounts[m]; jk++)
               // Then iterate over the neurons in layer m+1
               for (int ij = 0; ij < layerCounts[m + 1]; ij++)
               {
                  double randomValue = random(minRandomWeight,maxRandomWeight);
                  // If the weights line has more stuff, read it
                  if (weightsLine.hasMoreTokens())
                     weights[m][jk][ij] = parseDouble(weightsLine.nextToken(),randomValue);
                  else // Else default to a random double value in the range [low,high)
                     weights[m][jk][ij] = randomValue;
               }
         }
      }
      catch (IOException ioException)
      {
         ioException.printStackTrace();
         throw new RuntimeException("The weights file is not formatted properly");
      }
   }

   /**
    * Generate a number uniformly at random in the interval [low,high).
    *
    * This method takes 2 parameters, low and high, which give the bounds
    * for the random number generation.
    *
    * If low > high, then the number is in the range (high,low] instead.
    * (But the number returned is still properly "random" in about the same range).
    *
    * @param low  The lower bound of the random number generation
    * @param high The upper bound of the random number generation
    *
    * @return     A random double in the range [low,high)
    */
   private double random(double low, double high)
   {
      return (high - low) * Math.random() + low;
   }

   /**
    * This method can be called by the network handler class in order to
    * have the pdp network read the input values stored in the given file.
    *
    * This method takes the single parameter value inputsFile,
    * which has all of the inputs for the network stored
    * in a specific ordering and organization within the file.
    *
    * @param inputsFile       The file which holds all of the inputs for the network
    * @param numTestCases     The number of test cases in the training set
    *
    * @throws RuntimeException This method throws a runtime exception if anything
    *                          goes wrong during the file-reading process.
    *                          This method also prints out the stack trace
    *                          of the original error.
    */
   private double[][] readInputs(File inputsFile, int numTestCases)
   {
      try
      {
         // BufferedReader can read the inputsFile file
         BufferedReader in = new BufferedReader(new FileReader(inputsFile));

         // There are numTestCases total
         double[][] inputs = new double[numTestCases][layerCounts[0]];

         // Iterate over all the test cases
         for (int iterator = 0; iterator < numTestCases; iterator++)
         {
            // Make sure the next line is not null
            String textLine = in.readLine();
            if (textLine == null)
               textLine = "";

            // Make a StringTokenizer to read the line
            StringTokenizer inputsLine = new StringTokenizer(textLine);

            for (int inputIndex = 0; inputIndex < layerCounts[0]; inputIndex++)
               // If the inputs line ran out, use 0 (default double value)
               if (inputsLine.hasMoreTokens())
                  // Else read from the inputs line
                  inputs[iterator][inputIndex] = parseDouble(inputsLine.nextToken(),0.0);
         }

         return inputs;
      }
      catch (IOException ioException)
      {
         ioException.printStackTrace();
         throw new RuntimeException("The inputs file is not formatted properly");
      }
   }

   /**
    * This method can be called by the network handler class in order to
    * have the pdp network read the output values stored in the given file.
    *
    * This method takes the single parameter value outputsFile,
    * which has all of the outputs for the network stored
    * in a specific ordering and organization within the file.
    *
    * @param outputsFile       The file which holds all of the outputs for the network
    * @param numTestCases     The number of test cases in the training set
    *
    * @throws RuntimeException This method throws a runtime exception if anything
    *                          goes wrong during the file-reading process.
    *                          This method also prints out the stack trace
    *                          of the original error.
    */
   private double[][] readOutputs(File outputsFile, int numTestCases)
   {
      try
      {
         // BufferedReader can read the inputsFile file
         BufferedReader in = new BufferedReader(new FileReader(outputsFile));

         // There are numTestCases total
         int numOutputs = layerCounts[outputLayer];
         double[][] outputs = new double[numTestCases][numOutputs];

         // Iterate over all the test cases
         for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
         {
            // Make sure the next line is not null
            String textLine = in.readLine();
            if (textLine == null)
               textLine = "";

            StringTokenizer outputsLine = new StringTokenizer(textLine);

            for (int outputIndex = 0; outputIndex < numOutputs; outputIndex++)
               // If the outputs line ran out, use 0 (default double value)
               if (outputsLine.hasMoreTokens())
                  // Else read from the outputs line
                  outputs[caseIndex][outputIndex] = parseDouble(outputsLine.nextToken(),0.0);
         }

         return outputs;
      }
      catch (IOException ioException)
      {
         ioException.printStackTrace();
         throw new RuntimeException("The output file is not formatted properly");
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
   private double parseDouble(String nextToken, double defaultValue)
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
      catch (NumberFormatException numberFormatException)
      {
         return defaultValue;
      }
   }

   /**
    * This method can be called by the network handler class in order to
    * have the pdp network read the inputs values stored in the given file.
    *
    * This method takes the single parameter value inputsFile,
    * which has all of the inputs for the network stored
    * in a specific ordering and organization within the file.
    *
    * @param inputsFile        The file which holds all of the inputs for the network
    * @param numTestCases      The number of test cases in the training set
    *
    * @return A 2D array of doubles, where each row represents a new test case. The
    *         array rows are sorted in the order that the inputs are given. Each row
    *         of the array will have as many elements as output neurons in the network.
    */
   double[][] runNetwork(File inputsFile, int numTestCases)
   {
      // Get the inputs from the file
      double[][] inputs = readInputs(inputsFile, numTestCases);

      int numOutputs = layerCounts[outputLayer];

      // Run the network on the inputs
      double[][] outputs = new double[numTestCases][numOutputs];
      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         double[] output = runNetworkOnInputs(inputs[testCase]);
         for (int outputIndex = 0; outputIndex < numOutputs; outputIndex++)
            outputs[testCase][outputIndex] = output[outputIndex];
      }
      return outputs;
   }

   /**
    * This method runs the network on the given inputs, and
    * it returns the output of the network on those inputs.
    *
    * This method takes exactly 1 parameter, a 1D array of doubles inputs,
    * which represents the inputs on which to run the network.
    *
    * @param inputs The inputs on which to run the network
    *
    * @return A 1D array of doubles which represents the
    *         output of the network for the given set of inputs
    */
   private double[] runNetworkOnInputs(double[] inputs)
   {
      // Put the input values into the network
      activations[0] = unboundedActivations[0] = inputs;

      // Calculate the activation values for all activation layers
      for (int layer = 1; layer < numNeuronLayers; layer++)
      {
         // Get index of the previous layer
         int prevLayer = layer - 1;

         // Iterate over all the neurons in the layer with the given layer
         for (int layerElementIndex = 0; layerElementIndex < layerCounts[layer]; layerElementIndex++)
         {
            // The activation value of neuron indexed layerElementIndex is calculated - Currently Dot Product
            unboundedActivations[layer][layerElementIndex] = 0;
            for (int index = 0; index < layerCounts[prevLayer]; index++)
            {
               double activation = activations[prevLayer][index] * weights[prevLayer][index][layerElementIndex];
               unboundedActivations[layer][layerElementIndex] += activation;
            }

            // Apply the threshold function
            activations[layer][layerElementIndex] = thresholdFunction(unboundedActivations[layer][layerElementIndex]);
         }
      }

      // Return the 1D array of outputs
      return activations[outputLayer];
   }

   /**
    * This method trains the network.
    * It takes in 2 files which represent the input and output files. It also takes in
    * a single integer representing the number of test cases in the training set.
    *
    * The method reads the inputs and outputs from the file and begins training.
    * It runs until it hits the max number of iterations, it hits the minimum error,
    * or lambda drops below a minimum threshold.
    *
    * This method utilizes gradient descent and back-propagation to reach a minimum error.
    *
    * @param inputsFile    The inputs file
    * @param outputsFile   The outputs file
    * @param numTestCases  The number of test cases in the training set
    */
   void trainNetwork(File inputsFile, File outputsFile, int numTestCases)
   {
      double[][] inputs = readInputs(inputsFile, numTestCases);
      double[][] outputs = readOutputs(outputsFile, numTestCases);
      if (inputs.length != outputs.length || inputs.length != numTestCases)
         throw new IllegalStateException("input and output files don't hold the same number of cases");

      boolean continueTraining = true;
      iterationCounter = 0;

      // The error difference for this test case
      double[] errorDiff = omega[outputLayer];

      // Declare all variables outside all loops
      double psi, caseError, newCaseError, newErrorDiff;
      double[] calculatedOutputs, newCalculatedOutputs;

      // TIMING
//      int pingInterval = Math.min(1000, maximumIterationCount / 10);
//      long startTime = System.nanoTime(), lastPingTime = startTime;

      while (continueTraining)
      {
         maximumTestCaseError = 0.0;

         for (int testCaseIndex = 0; testCaseIndex < numTestCases; testCaseIndex++)
         {
            // Run network on test case to store activation values into array - Get the outputs
            calculatedOutputs = runNetworkOnInputs(inputs[testCaseIndex]);
            for (int outputIndex = 0; outputIndex < layerCounts[outputLayer]; outputIndex++)
               omega[outputLayer][outputIndex] = outputs[testCaseIndex][outputIndex] - calculatedOutputs[outputIndex];

            // Get current case error
            caseError = 0.0;
            for (int i = 0; i < layerCounts[outputLayer]; i++) // Output Layer
               caseError += errorDiff[i] * errorDiff[i] / 2.0;

            // BACK-PROPAGATION!!
            for (int layerIndex = outputLayer - 1; layerIndex >= 0; layerIndex--)
            {
               int rightLayer = layerIndex + 1;
               for (int rightIndex = 0; rightIndex < layerCounts[rightLayer]; rightIndex++)
               {
                  // Get psi
                  psi = omega[rightLayer][rightIndex] * thresholdFunctionDeriv(unboundedActivations[rightLayer][rightIndex]);
                  // Clear omega array
                  omega[rightLayer][rightIndex] = 0;

                  for (int leftIndex = 0; leftIndex < layerCounts[layerIndex]; leftIndex++)
                  {
                     omega[layerIndex][leftIndex] += psi * weights[layerIndex][leftIndex][rightIndex];
                     deltaWeights[layerIndex][leftIndex][rightIndex] = psi * lambda * activations[layerIndex][leftIndex];
                     weights[layerIndex][leftIndex][rightIndex] += deltaWeights[layerIndex][leftIndex][rightIndex];
                  }
               }
            }

            // Get new case error
            newCalculatedOutputs = runNetworkOnInputs(inputs[testCaseIndex]);
            newCaseError = 0.0;
            for (int i = 0; i < layerCounts[outputLayer]; i++) // Output Layer
            {
               newErrorDiff = outputs[testCaseIndex][i] - newCalculatedOutputs[i];
               newCaseError += newErrorDiff * newErrorDiff / 2.0;
            }

            if (newCaseError < caseError)
            {
               // Cap lambda (learning factor) to lambdaMaxCap
               if (lambda < lambdaMaxCap)
                  lambda *= lambdaChange;

               if (maximumTestCaseError < newCaseError)
                  maximumTestCaseError = newCaseError;
            }
            else
            {
               lambda /= lambdaChange;
               for (int m = 0; m < numWeightLayers; m++)
                  for (int jk = 0; jk < layerCounts[m]; jk++)
                     for (int ij = 0; ij < layerCounts[m+1]; ij++)
                        weights[m][jk][ij] -= deltaWeights[m][jk][ij];

               if (maximumTestCaseError < caseError)
                  maximumTestCaseError = caseError;
            }
         }

         if (lambda < lambdaMinCap)
         {
            System.out.println("Lambda went below Minimum Lambda Capacity: " + lambda + " < " + lambdaMinCap);
            continueTraining = false;
         }

         if (maximumTestCaseError < minimumError)
         {
            System.out.println("Maximum Test Case Error went below Minimum Error Success Threshold: " +
                  maximumTestCaseError + " < " + minimumError);

            continueTraining = false;
         }

         iterationCounter++;
         if (iterationCounter >= maximumIterationCount)
         {
            System.out.println("Training Iterations hit Iteration Capacity: " +
                    iterationCounter + " >= " + maximumIterationCount);

            continueTraining = false;
         }

         // TIMING
//         if (continueTraining && iterationCounter % pingInterval == 0)
//         {
//            long pingTime = System.nanoTime();
//
//            String message = "TIME ELAPSED for Iterations " + (iterationCounter - pingInterval) + " to " +
//                  iterationCounter +  ": " + ((double) (pingTime - lastPingTime) / 1000000000.0);
//
//            System.out.println(message);
//            lastPingTime = System.nanoTime();
//         }
      }

      System.out.println();
      System.out.println("Total Iterations: " + iterationCounter);

      // TIMING
//      long endTime = System.nanoTime();
//      System.out.println("TOTAL TIME ELAPSED: " + ((double) (endTime - startTime) / 1000000000.0));

      System.out.println();
      System.out.println("weights: CHECK WEIGHT DUMP");
      System.out.println("outputs: CHECK OUTPUT DUMP");

      System.out.println();
      System.out.println("errors: NOT SUPPORTED RIGHT NOW");
      System.out.println("lambda: " + lambda);
   }

   /**
    * This method is the neuron threshold function
    * for this pdp network. It is designed to limit
    * the values passed through the network in order
    * to prevent the escalation of those values.
    * However, for simpler networks, this threshold
    * function can be omitted by making this method
    * simply return its parameter (input) directly.
    *
    * This method takes one parameter (input),
    * and applies a threshold function to
    * that double value in order to bound it
    * between some values (usually 0 to 1 or -1 to 1).
    *
    * @param neuronInput The input to be bounded
    *
    * @return The bounded version of the input
    */
   private double thresholdFunction(double neuronInput)
   {
      // f(x) = x
      //return neuronInput;

      // f(x) = Sigmoid Function
      return 1.0 / (1.0 + Math.exp(-neuronInput));

      // f(x) = Gaussian Distribution Function
      //return Math.exp(-neuronInput * neuronInput);
   }

   /**
    * This method is the derivative of the neuron
    * threshold function for this pdp network. It
    * is used in the error calculations for the
    * neurons of the network.
    *
    * This method takes one parameter (input),
    * and applies the derivative of the threshold
    * function to that double value so that the
    * value can be used for the partial derivative
    * calculations later in the back propagation process.
    *
    * @param neuronInput The input to be processed in the
    *                    derivative of the threshold function
    *
    * @return The threshold function's derivative
    *         evaluated at the input double
    */
   private double thresholdFunctionDeriv(double neuronInput)
   {
      // f(x) = x
      //return 1;

      // f(x) = Sigmoid Function
      double sigmoidValue = thresholdFunction(neuronInput);
      return sigmoidValue * (1.0 - sigmoidValue);

      // f(x) = Gaussian Distribution Function
      //return -2.0 * neuronInput * neuronThresholdFunction(neuronInput);
   }
}