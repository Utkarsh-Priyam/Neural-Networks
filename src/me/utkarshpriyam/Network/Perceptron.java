package me.utkarshpriyam.Network;

import java.io.*;
import java.util.Arrays;

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
 *
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
    private static final double LAMBDA = 0.1;

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
        // Throw an IllegalArgumentException if the array passed
        // doesn't have a length of at least 2 (for the number of
        // neurons in the input and output layers of the pdp network)
        if (layerCounts.length < 2)
            throw new IllegalArgumentException("not enough layers in network");

        // Store the passed array of the counts of the number of
        // neurons per layer and shorten the array to exclude the
        // counts of the input and output layers (for processing)
        this.layerCounts = layerCounts;
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
        // Compact all the layers data into one single array
        layerCounts = new int[hiddenLayersCount.length + 2];
        layerCounts[0] = numInputs;
        System.arraycopy(hiddenLayersCount, 0, layerCounts, 1, hiddenLayersCount.length);
        layerCounts[layerCounts.length-1] = numOutputs;

        // Generate the neuron and edge arrays
        generateNeuronsAndEdgesArrays(numInputs,hiddenLayersCount,numOutputs);
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
            // The file is expected to have as many rows
            // with doubles as the network has edge layers
            for (int m = 0; m < weights.length; m++)
            {
                // Split the row's text at the spaces
                String[] weightsLine = w.readLine().split(" ");

                // Iterate over all the edges in the layer with index m
                // The file is expected to have enough doubles per line of
                // text in order to fill up the entire array of weights

                // Iterate over the neurons in layer m first
                for (int jk = 0; jk < layerCounts[m]; jk++)
                    // Then iterate over the neurons in layer m+1
                    for (int ij = 0; ij < layerCounts[m+1]; ij++)
                        // For each of these edges read the weight from the file
                        weights[m][jk][ij] = Double.parseDouble(weightsLine[jk * layerCounts[m+1] + ij]);
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
                // Split the row of text in the file by spaces
                // (number of inputs in row should equal expected amount)
                String[] inputsLine = in.readLine().split(" ");
                for (int inputIndex = 0; inputIndex < layerCounts[0]; inputIndex++)
                    inputs[iterator][inputIndex] = Double.parseDouble(inputsLine[inputIndex]);
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
     *
     * @throws RuntimeException This method throws a runtime exception if anything
     *                          goes wrong during input-reading or processing.
     *                          This method also prints out the stack trace
     *                          of the original error.
     *                          An exception is thrown if the given input file is not found,
     *                          the file is not properly formatted, a given weight is not a
     *                          double, or not enough inputs are given in the inputs file.
     */
    protected double[][] runNetwork(File inputsFile)
    {
        // Get the inputs from the file
        double[][] inputs = readInputs(inputsFile);

        // A 2D array is created with enough rows to store all the test cases individually
        double[][] outputs = new double[inputs.length][];

        // Iterate over all the test cases
        for (int iterator = 0; iterator < inputs.length; iterator++)
        {
            // Put the input values into the network
            activations[0] = inputs[iterator];

            // For each layer 1 to activations.length-1, calculate
            // the activation values for the neurons of the layers
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
        return neuronInput;
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
