package me.utkarshpriyam.Network;

import java.awt.*;
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
 * WORK IN PROGRESS:
 * In Training Mode (as the name suggests, to train the network):
 *   - The network will take in both the inputs and the expected outputs
 *   - The network will automatically calculate its error and
 *     use the method of gradient descent to adjust its weights
 *     in order to reduce that aforementioned error
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
public class Perceptron {
   /**
    * This boolean constant dictates whether to instantiate the
    * neuron and edge arrays as a full block (which has wasted space)
    * or as a ragged array (no wasted space)
    */
   private static final boolean GENERATE_RAGGED_ARRAYS = false;

   /**
    * This double value is the learning factor for this pdp network.
    * It is the step size that dictates how "fast" or "slowly" the
    * network adjusts its weights (whether it takes big or small
    * steps in the "downhill direction" (method of gradient descent).
    */
   private double lambda = 0.1;

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
   private double[][] activations;
   private double[][][] weights;

   /**
    * This is the first constructor for the Perceptron class.
    * It takes in a single array that contains the counts
    * of the number of neurons in each layer of the network.
    *
    * @param layerCounts The array which contains the number of
    *                    neurons in each layer of the network
    *
    * @throws IllegalArgumentException This method throws an IllegalArgumentException
    *                                  if the parameter passed does not have at least
    *                                  2 values (for the input and output layers).
    */
   public Perceptron(int[] layerCounts)
   {
      // Throw an IllegalArgumentException if not enough layers (2) are passed
      if (layerCounts.length < 2)
         throw new IllegalArgumentException("not enough layers in network");

      // Adjust the inputs to ensure that every layer has at least 1 neuron
      layerCounts = setMinimumAllowed(layerCounts,1);

      // Store the layers array
      this.layerCounts = layerCounts;
      // Shorten the layer counts array to exclude the input and output counts
      int[] innerLayerCounts = Arrays.copyOfRange(layerCounts,1,layerCounts.length - 1);

      // Generate the neuron and edge arrays
      generateNeuronsAndEdgesArrays(layerCounts[0],innerLayerCounts,layerCounts[layerCounts.length-1]);
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
    */
   public Perceptron(int numInputs, int[] hiddenLayersCount, int numOutputs)
   {
      // Adjust the inputs to ensure that every layer has at least 1 neuron
      if (numInputs < 1)
         numInputs = 1;
      if (numOutputs < 1)
         numOutputs = 1;
      hiddenLayersCount = setMinimumAllowed(hiddenLayersCount,1);

      // Compact all the layers data into one single array
      layerCounts = new int[hiddenLayersCount.length + 2];
      layerCounts[0] = numInputs;
      System.arraycopy(hiddenLayersCount, 0, layerCounts, 1, hiddenLayersCount.length);
      layerCounts[layerCounts.length-1] = numOutputs;

      // Generate the neuron and edge arrays
      generateNeuronsAndEdgesArrays(numInputs,hiddenLayersCount,numOutputs);
   }

   /**
    * Goes through the given array and ensures that all values of the array
    * are at least as large as the given input integer
    * @param hiddenLayersCount The array to read and modify
    * @param minimumValue      The minimum allowed value
    *
    * @return The adjusted array
    */
   private int[] setMinimumAllowed(int[] hiddenLayersCount, int minimumValue)
   {
      int[] newArray = new int[hiddenLayersCount.length];
      for (int i = 0; i < hiddenLayersCount.length; i++)
         newArray[i] = Math.max(hiddenLayersCount[i], minimumValue);
      return newArray;
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
      activations = new double[2 + hiddenLayersCount.length][];

      // Set Input Neurons Array length
      activations[0] = new double[numInputs];
      // Set Hidden Neurons Array length
      for (int i = 1; i < activations.length - 1; i++)
         activations[i] = new double[hiddenLayersCount[i]];
      // Set Output Neurons Array length
      activations[activations.length - 1] = new double[numOutputs];

      // Generate Edges Array - Total Layers = numNeurons - 1
      weights = new double[activations.length - 1][][];

      // Generate Second and Third Array Dimensions: Length = numNeurons in prev layer, next layer
      for (int m = 0; m < weights.length; m++)
         weights[m] = new double[activations[m].length][activations[m+1].length];
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
      // The number will serve as the 2nd, 2nd and 3rd dimensions
      // for the neurons, edges arrays
      int maxNumNeurons = Math.max(numInputs,numOutputs);
      for (int numNeurons: hiddenLayersCount)
         if (maxNumNeurons < numNeurons)
            maxNumNeurons = numNeurons;

      // Generate both arrays
      activations = new double[2 + hiddenLayersCount.length][maxNumNeurons];
      weights = new double[activations.length - 1][maxNumNeurons][maxNumNeurons];
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
    *
    * @throws RuntimeException This method throws a runtime exception if anything
    *                          goes wrong during the file-reading process.
    *                          This method also prints out the stack trace
    *                          of the original error.
    */
   void readWeights(File weightsFile)
   {
      try
      {
         // BufferedReader w can read all the weights out of the weightsFile file
         BufferedReader w = new BufferedReader(new FileReader(weightsFile));

         // Iterate through all the different weights layers
         for (int m = 0; m < weights.length; m++)
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
               for (int ij = 0; ij < layerCounts[m+1]; ij++)
                  // If the weights line ran out, use 0 (default double value)
                  if (weightsLine.hasMoreTokens())
                     // Else read from the weights line
                     weights[m][jk][ij] = parseDouble(weightsLine.nextToken());
         }
      }
      catch (IOException ioException)
      {
         ioException.printStackTrace();
         throw new RuntimeException("The weights file is not formatted properly");
      }
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
    *
    * @throws RuntimeException This method throws a runtime exception if anything
    *                          goes wrong during the file-reading process.
    *                          This method also prints out the stack trace
    *                          of the original error.
    */
   private double[][] readInputs(File inputsFile)
   {
      try
      {
         // BufferedReader can read the inputsFile file
         BufferedReader in = new BufferedReader(new FileReader(inputsFile));

         // The file is formatted so the first line tells how many test cases there are
         int numTestCases = Integer.parseInt(in.readLine());
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
                  inputs[iterator][inputIndex] = parseDouble(inputsLine.nextToken());
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
    *
    * @throws RuntimeException This method throws a runtime exception if anything
    *                          goes wrong during the file-reading process.
    *                          This method also prints out the stack trace
    *                          of the original error.
    */
   private double[][] readOutputs(File outputsFile)
   {
      try
      {
         // BufferedReader can read the inputsFile file
         BufferedReader in = new BufferedReader(new FileReader(outputsFile));

         // The file is formatted so the first line tells how many test cases there are
         int numTestCases = Integer.parseInt(in.readLine());
         int numOutputs = layerCounts[layerCounts.length-1];
         double[][] outputs = new double[numTestCases][numOutputs];

         // Iterate over all the test cases
         for (int caseIterator = 0; caseIterator < numTestCases; caseIterator++)
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
                  outputs[caseIterator][outputIndex] = parseDouble(outputsLine.nextToken());
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
    * If the token is not a double, then it just returns 0 by default.
    *
    * This method takes one String parameter nextToken
    *
    * @param nextToken The token to parse
    *
    * @return The parsed double, or 0 if the token cannot be parsed
    */
   private double parseDouble(String nextToken)
   {
      try
      {
         return Double.parseDouble(nextToken);
      }
      catch (NumberFormatException numberFormatException)
      {
         return 0;
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
    *
    * @return A 2D array of doubles, where each row represents a new test case. The
    *         array rows are sorted in the order that the inputs are given. Each row
    *         of the array will have as many elements as output neurons in the network.
    */
   protected double[][] runNetwork(File inputsFile)
   {
      // Get the inputs from the file
      double[][] inputs = readInputs(inputsFile);

      // Run the network on the inputs
      return runNetworkOnInputs(inputs);
   }

   /**
    * This method runs the network on the given inputs, and
    * it returns the output of the network on those inputs.
    *
    * This method takes exactly 1 parameter, a 2D array of doubles inputs,
    * which represents the inputs on which to run the network.
    *
    * @param inputs The inputs on which to run the network
    *
    * @return An array of doubles which represents the
    *         output of the network for the given set of inputs
    */
   private double[][] runNetworkOnInputs(double[][] inputs)
   {
      // A 2D array is created with enough rows to store all the test cases individually
      double[][] outputs = new double[inputs.length][];

      // Iterate over all the test cases
      for (int iterator = 0; iterator < inputs.length; iterator++)
      {
         // Put the input values into the network
         activations[0] = inputs[iterator];

         // Calculate the activation values for all activation layers
         for (int n = 1; n < activations.length; n++)
            calculateActivations(n);

         // Index of output layer
         int outputLayerIndex = activations.length - 1;

         // Store the values of the output neurons into the 2D outputs array
         double[] activatedNeurons = activations[outputLayerIndex];
         int countNeurons = layerCounts[outputLayerIndex];
         outputs[iterator] = Arrays.copyOfRange(activatedNeurons, 0, countNeurons);
      }
      // Return the 2D outputs array
      return outputs;
   }

   /**
    * This method trains the network
    *
    * ADD JAVADOC FOR THIS METHOD
    *
    * TODO: (from 9/16/19)
    *
    * @param inputsFile
    * @param outputsFile
    */
   protected void trainNetwork(File inputsFile, File outputsFile)
   {
      double[][] inputs = readInputs(inputsFile);
      double[][] outputs = readOutputs(outputsFile);
      if (inputs.length != outputs.length)
         throw new IllegalStateException("input and output files don't hold the same number of cases");

      System.out.println("inputs: " + Arrays.deepToString(inputs));
      System.out.println("outputs: " + Arrays.deepToString(outputs));

      for (int i = 0; i < 100; i++)
         runTrainingStep(inputs,outputs);

      System.out.println();
      System.out.println("weights: " + Arrays.deepToString(weights));
      System.out.println("lambda: " + lambda);
   }

   private void runTrainingStep(double[][] inputs, double[][] outputs)
   {
      if (layerCounts.length != 3)
         throw new RuntimeException("training currently only works for M-N-1 networks");

      double[][] calculatedOutputs = runNetworkOnInputs(inputs);
      int numTestCases = calculatedOutputs.length;
      for (int testCaseIterator = 0; testCaseIterator < numTestCases; testCaseIterator++)
      {
         /*
         System.out.println();
         System.out.println("test case: " + testCaseIterator);
         System.out.println(Arrays.deepToString(weights));
          */

         // The weights adjustment array
         double[][][] weightAdjustments = new double[weights.length][weights[0].length][weights[0][0].length];
         // The error difference for this test case
         double errorDiff = outputs[testCaseIterator][0] - calculatedOutputs[testCaseIterator][0];

         /*
         System.out.println("case error: " + errorDiff);
         System.out.println("lambda: " + lambda);
          */

         for (int j = 0; j < layerCounts[1]; j++) // Middle Layer
            for (int i = 0; i < layerCounts[2]; i++) // Output Layer
            {
               // System.out.println("(m,j,i) = (1," + j + "," + i + ")");

               double[] prevNeurons = Arrays.copyOfRange(activations[1], 0, layerCounts[1]);
               double[] prevWeights = new double[prevNeurons.length];
               for (int ijkPrevLayer = 0; ijkPrevLayer < layerCounts[1]; ijkPrevLayer++)
                  prevWeights[ijkPrevLayer] = weights[1][ijkPrevLayer][i];

               double activationValueUnbounded = neuronPrevLayerCombinerFunction(prevNeurons, prevWeights);
               //System.out.println("output val: " + activationValueUnbounded);
               // These 2 lines handle this derivative --> d F_i/d W_abc
               double adjustment = neuronThresholdFunctionDeriv(activationValueUnbounded);
               //System.out.println("threshold deriv: " + adjustment);
               adjustment *= neuronPrevLayerCombinerFunctionDeriv(prevNeurons, prevWeights, j, true);
               //System.out.println("combiner funct deriv: " + adjustment);
               // Multiply by learning factor and error diff to get delta W
               adjustment *= lambda * errorDiff;
               // Store delta W in array
               weightAdjustments[1][j][i] = adjustment;
               //System.out.println("total adjustment: " + adjustment);
            }

         for (int k = 0; k < layerCounts[0]; k++) // Input Layer
            for (int j = 0; j < layerCounts[1]; j++) // Middle Layer
            {
               //System.out.println("(m,k,j) = (0," + k + "," + j + ")");

               double[] prevNeurons = Arrays.copyOfRange(activations[0], 0, layerCounts[0]);
               double[] prevWeights = new double[prevNeurons.length];
               for (int ijkPrevLayer = 0; ijkPrevLayer < layerCounts[0]; ijkPrevLayer++)
                  prevWeights[ijkPrevLayer] = weights[0][ijkPrevLayer][j];

               double[] nextNeurons = Arrays.copyOfRange(activations[1], 0, layerCounts[1]);
               double[] nextWeights = new double[nextNeurons.length];
               for (int ijkNextLayer = 0; ijkNextLayer < layerCounts[1]; ijkNextLayer++)
                  nextWeights[ijkNextLayer] = weights[1][ijkNextLayer][0];

               double activationValueUnbounded = neuronPrevLayerCombinerFunction(prevNeurons, prevWeights);
               double outputValueUnbounded = neuronPrevLayerCombinerFunction(nextNeurons, nextWeights);
               // These 2 lines handle this derivative --> d f(h_j)/d W_abc
               double adjustment = neuronThresholdFunctionDeriv(outputValueUnbounded);
               //System.out.println("step 1: " + adjustment);
               adjustment *= neuronPrevLayerCombinerFunctionDeriv(nextNeurons, nextWeights, j, false);
               //System.out.println("step 2: " + adjustment);
               //System.out.println("next weights: " + Arrays.toString(nextWeights));

               // These 2 lines handle this derivative --> d F_i/d W_abc
               adjustment *= neuronThresholdFunctionDeriv(activationValueUnbounded);
               //System.out.println("step 3: " + adjustment);
               adjustment *= neuronPrevLayerCombinerFunctionDeriv(prevNeurons, prevWeights, k, true);
               //System.out.println("step 4: " + adjustment);

               // Multiply by learning factor and error diff to get delta W
               adjustment *= lambda * errorDiff;
               //System.out.println("step 5: " + adjustment);
               // Store delta W in array
               weightAdjustments[0][k][j] = adjustment;

               //System.out.println("adjustment: " + adjustment);
            }

         // Scaled, Positive Error for this case
         double caseError = errorCalculator(outputs,calculatedOutputs)[0];

         for (int m = 0; m < weights.length; m++)
            for (int jk = 0; jk < layerCounts[m]; jk++)
               for (int ij = 0; ij < layerCounts[m+1]; ij++)
                  weights[m][jk][ij] += weightAdjustments[m][jk][ij];

         double[][] newCalculatedOutputs = runNetworkOnInputs(inputs);
         double newCaseError = errorCalculator(outputs,newCalculatedOutputs)[0];

         if (newCaseError <= caseError)
         {
            lambda *= 2.0;
            calculatedOutputs = newCalculatedOutputs;
         }
         else
         {
            lambda /= 2.0;
            for (int m = 0; m < weights.length; m++)
               for (int jk = 0; jk < layerCounts[m]; jk++)
                  for (int ij = 0; ij < layerCounts[m+1]; ij++)
                     weights[m][jk][ij] -= weightAdjustments[m][jk][ij];
         }
      }
   }

   /**
    *
    */
   private double[] errorCalculator(double[][] expected, double[][] calculated)
   {
      if (expected.length != calculated.length || expected.length == 0)
         throw new IllegalArgumentException("The expected and calculated arrays must both have the same " +
               "non-zero number of test cases");

      if (expected[0].length != calculated[0].length)
         throw new IllegalArgumentException("The expected and calculated arrays must both have " +
               "the same number of output neuron values");

      double[] errors = new double[expected[0].length];
      for (int outputIterator = 0; outputIterator < errors.length; outputIterator++)
      {
         double error = 0;

         for (int testCaseIterator = 0; testCaseIterator < expected.length; testCaseIterator++)
         {
            double expectedValue = expected[testCaseIterator][outputIterator];
            double calculatedValue = calculated[testCaseIterator][outputIterator];

            double testCaseError = expectedValue - calculatedValue;

            error += testCaseError * testCaseError;
         }

         errors[outputIterator] = error/2;
      }

      return errors;
   }

   /**
    * This method calculates the activation of
    * all the neurons in the given layer.
    *
    * This method takes the single parameter layer,
    * which tells which layer index to evaluate.
    *
    * @param layer The index of the layer to evaluate.
    *              The layers are indexed from left to
    *              right, with the input layer as 0 and
    *              the output layer as layerCounts.length-1
    */
   private void calculateActivations(int layer)
   {
      // Get index of the previous layer
      int prevLayer = layer - 1;

      // Iterate over all the neurons in the layer with the given layer
      for (int ijk = 0; ijk < layerCounts[layer]; ijk++)
      {
         // All the neurons from the previous layer, stored in an array
         double[] prevNeurons = Arrays.copyOfRange(activations[prevLayer],0,layerCounts[prevLayer]);
         // All the weights from the previous layer, stored in an array
         double[] prevWeights = new double[prevNeurons.length];

         for (int ijkPrevLayer = 0; ijkPrevLayer < layerCounts[prevLayer]; ijkPrevLayer++)
            prevWeights[ijkPrevLayer] = weights[prevLayer][ijkPrevLayer][ijk];

         // The activation value of neuron indexed ijk is calculated
         activations[layer][ijk] = neuronActivationValueCalculator(prevNeurons,prevWeights);
      }
   }

   /**
    * This method calculates how all the neurons
    * from the previous layer are combined with their weights
    * in order to for the input for the next layer's neuron.
    * It then applies the threshold function to that input
    * in order to calculate the activation value for the
    * currently processing neuron.
    *
    * This method takes two parameters (neurons and weights),
    * two arrays which hold the neurons of the previous layer
    * and the weights of the edges which connect those neurons
    * to the target neuron whose input is being calculated.
    *
    * @param neurons   The neurons of the previous layer
    * @param weights The weights of the edges which connect
    *                the neurons (from neurons) to the target neuron
    *
    * @return This method returns the neuron's input bounded by the
    *         threshold function. It calculates the neuron's input
    *         via the neuronPrevLayerCombinerFunction(...) method.
    */
   private double neuronActivationValueCalculator(double[] neurons, double[] weights)
   {
      double neuronInput = neuronPrevLayerCombinerFunction(neurons,weights);
      return neuronThresholdFunction(neuronInput);
   }

   /**
    * This method calculates how all the neurons
    * from the previous layer are combined with their weights
    * in order to for the input for the next layer's neuron.
    *
    * This method takes two parameters (neurons and weights),
    * two arrays which hold the neurons of the previous layer
    * and the weights of the edges which connect those neurons
    * to the target neuron whose input is being calculated.
    *
    * @param neurons   The neurons of the previous layer
    * @param weights The weights of the edges which connect
    *                the neurons (from neurons) to the target neuron
    *
    * @return This method returns a combination of the neurons
    *         and weights arrays which a neuron can use as its
    *         input for the threshold function.
    *
    * @throws IllegalArgumentException This method throws an IllegalArgumentException
    *                                  if the neurons and weights arrays passed as
    *                                  parameters don't have the same length.
    */
   private double neuronPrevLayerCombinerFunction(double[] neurons, double[] weights)
   {
      // if the neurons and weights arrays passed as parameters don't have the same length
      if (neurons.length != weights.length)
         throw new IllegalArgumentException("neurons and weights arrays don't have equal lengths");

      // Currently Dot Product
      double input = 0;

      for (int index = 0; index < neurons.length; index++)
         input += neurons[index] * weights[index];

      return input;
   }

   /**
    * This method calculates the derivative of how all the neurons
    * from the previous layer are combined with their weights
    * in order to for the input for the next layer's neuron.
    *
    * This method takes four parameters (neurons, weights,
    * index, and isWeight). neurons and weights are two arrays
    * which hold the neurons of the previous layer and
    * the weights of the edges which connect those neurons
    * to the target neuron whose input is being calculated.
    * index refers to the index of the weight or neuron with
    * respect to which this method is taking the derivative.
    * isWeight refers to whether the index is for the weight
    * or the neuron at with the given index.
    *
    * @param neurons    The neurons of the previous layer
    * @param weights  The weights of the edges which connect
    *                 the neurons (from neurons) to the target neuron
    * @param index    The index of the element with respect to which
    *                 this method is taking the derivative
    * @param isWeight This boolean is true if the element wrt which
    *                 the derivative is being taken is a weight,
    *                 and false if the element is a neuron.
    *                 The reason functionality for both is included
    *                 is so that this method can also be used for the
    *                 derivatives required for the edges present in the
    *                 inner layers (closer to input layer) of the network.
    *
    * @return This method returns the value of the derivative of the
    *         previous layer combiner function taken with respect to
    *         the element (either edge or neuron) at the given index.
    *
    * @throws IllegalArgumentException This method throws an IllegalArgumentException
    *                                  if the neurons and weights arrays passed as
    *                                  parameters don't have the same length.
    *                                  It also throws an IllegalArgumentException if
    *                                  the index passed is not a valid index in the
    *                                  neurons/weights array(s) passed as parameters.
    */
   private double neuronPrevLayerCombinerFunctionDeriv(double[] neurons, double[] weights, int index, boolean isWeight)
   {
      // if the neurons and weights arrays passed as parameters don't have the same length
      if (neurons.length != weights.length)
         throw new IllegalArgumentException("neurons and weights arrays don't have equal lengths");

      // if the index passed is not a valid index in the neurons/weights array(s)
      if (index < 0 || index >= neurons.length)
         throw new IllegalArgumentException("index given is not in the valid range for indices");

      /* For the dot product combination, the derivative returns the coefficient: */

      // if index isWeight: return neurons[index] (weights coefficient)
      if (isWeight)
         return neurons[index];
      // else (! isWeight): return weights[index] (neurons coefficient)
      return weights[index];
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
   private double neuronThresholdFunction(double neuronInput)
   {
      // Currently f(x) = x
      return neuronInput; // 1.0/(1.0 + Math.exp(-neuronInput)); // Sigmoid Function
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
   private double neuronThresholdFunctionDeriv(double neuronInput)
   {
      // Currently D(f(x)) = D(x) = 1
      return 1;
   }
}
