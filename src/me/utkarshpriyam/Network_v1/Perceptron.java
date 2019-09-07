package me.utkarshpriyam.Network_v1;

import java.io.*;
import java.util.Arrays;

/**
 * This is the Perceptron class.
 * It represents a pdp (parallel distributive perceptron).
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
 *   - The pdp will output the final raw information it calculates
 *   - The pdp will not take any predicted output values
 *      - As a result, the pdp will neither train its weights
 *        not return an error value
 *
 * In Training Mode (as the name suggests, to train the network):
 *   - The pdp will take in both the inputs and the expected outputs
 *   - The pdp will automatically calculate its error and
 *     use the method of gradient descent to adjust its weights
 *     in order to reduce that aforementioned error
 *
 * In Testing Mode (this is a blend of the previous two modes):
 *   - The pdp will take both the inputs and the expected outputs.
 *     However, as this is not a training exercise, the pdp will not
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
     */
    public Perceptron(int[] layerCounts) {
        if (layerCounts.length < 2)
            throw new IllegalArgumentException("not enough layers in network");
        this.layerCounts = layerCounts;
        int[] innerLayerCounts = Arrays.copyOfRange(layerCounts,1,layerCounts.length - 1);
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
    public Perceptron(int numInputs, int[] hiddenLayersCount, int numOutputs) {
        layerCounts = new int[hiddenLayersCount.length + 2];
        layerCounts[0] = numInputs;
        System.arraycopy(hiddenLayersCount, 0, layerCounts, 1, hiddenLayersCount.length);
        layerCounts[layerCounts.length-1] = numOutputs;

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
    private void generateNeuronsAndEdgesArrays(int numInputs, int[] hiddenLayersCount, int numOutputs) {
        if (GENERATE_RAGGED_ARRAYS)
            generateArraysRagged(numInputs,hiddenLayersCount,numOutputs);
        else
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
    private void generateArraysRagged(int numInputs, int[] hiddenLayersCount, int numOutputs) {
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
    private void generateArraysRegular(int numInputs, int[] hiddenLayersCount, int numOutputs) {
        // Count number maximum number of nodes in network
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
     *                          goes wrong during the weight-reading process.
     *                          This method also prints out the stack trace
     *                          of the original error.
     *                          An exception is thrown if the given weights file is not found,
     *                          the file is not properly formatted, a given weight is not a
     *                          double, or not enough weights are given in the weights file.
     */
    void setWeights(File weightsFile) {
        try
        {
            BufferedReader w = new BufferedReader(new FileReader(weightsFile));
            for (int m = 0; m < weights.length; m++)
            {
                String[] weightsLine = w.readLine().split(" ");
                for (int jk = 0; jk < layerCounts[m]; jk++)
                    for (int ij = 0; ij < layerCounts[m+1]; ij++)
                        weights[m][jk][ij] = Double.parseDouble(weightsLine[jk * layerCounts[m+1] + ij]);
            }
        }
        catch (FileNotFoundException fileNotFoundException)
        {
            fileNotFoundException.printStackTrace();
            throw new RuntimeException("Weights file not found");
        }
        catch (IOException ioException)
        {
            ioException.printStackTrace();
            throw new RuntimeException("The weights file is not formatted properly");
        }
        catch (NumberFormatException numberFormatException)
        {
            numberFormatException.printStackTrace();
            throw new RuntimeException("Given weight could not be converted to a double");
        }
        catch (ArrayIndexOutOfBoundsException arrayIndexOutOfBoundsException)
        {
            arrayIndexOutOfBoundsException.printStackTrace();
            throw new RuntimeException("Not enough weights given in file");
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
     * @throws RuntimeException This method throws a runtime exception if anything
     *                          goes wrong during input-reading or processing.
     *                          This method also prints out the stack trace
     *                          of the original error.
     *                          An exception is thrown if the given input file is not found,
     *                          the file is not properly formatted, a given weight is not a
     *                          double, or not enough inputs are given in the inputs file.
     */
    protected double[][] runNetwork(File inputsFile) {
        try
        {
            BufferedReader in = new BufferedReader(new FileReader(inputsFile));

            int numTestCases = Integer.parseInt(in.readLine());
            double[][] outputs = new double[numTestCases][];

            for (int iterator = 0; iterator < numTestCases; iterator++)
            {
                String[] inputsLine = in.readLine().split(" ");
                for (int k = 0; k < layerCounts[0]; k++)
                    activations[0][k] = Integer.parseInt(inputsLine[k]);

                for (int n = 1; n < activations.length; n++)
                    calculateActivations(n);

                int outputLayerIndex = activations.length - 1;

                outputs[iterator] = Arrays.copyOfRange(activations[outputLayerIndex], 0, layerCounts[outputLayerIndex]);
            }
            return outputs;
        }
        catch (FileNotFoundException fileNotFoundException)
        {
            fileNotFoundException.printStackTrace();
            throw new RuntimeException("Inputs file not found");
        }
        catch (IOException ioException)
        {
            ioException.printStackTrace();
            throw new RuntimeException("The inputs file is not formatted properly");
        }
        catch (NumberFormatException numberFormatException)
        {
            numberFormatException.printStackTrace();
            throw new RuntimeException("Given input could not be converted to an integer");
        }
        catch (ArrayIndexOutOfBoundsException arrayIndexOutOfBoundsException)
        {
            arrayIndexOutOfBoundsException.printStackTrace();
            throw new RuntimeException("Not enough inputs given in file");
        }
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
    private void calculateActivations(int layer) {
        int prevLayer = layer - 1;
        for (int ijk = 0; ijk < layerCounts[layer]; ijk++)
        {
            double[] prevNodes = Arrays.copyOfRange(activations[prevLayer],0,layerCounts[prevLayer]);
            double[] prevWeights = new double[prevNodes.length];

            for (int ijkPrevLayer = 0; ijkPrevLayer < layerCounts[prevLayer]; ijkPrevLayer++)
                prevWeights[ijkPrevLayer] = weights[prevLayer][ijkPrevLayer][ijk];

            double input = neuronInputCalculator(prevNodes,prevWeights);
            activations[layer][ijk] = neuronThresholdFunction(input);
        }
    }

    /**
     * This method calculates how all the nodes
     * from the previous layer are combined with their weights
     * in order to for the input for the next layer's node.
     *
     * This method takes two parameters (nodes and weights),
     * two arrays which hold the neurons of the previous layer
     * and the weights of the edges which connect those neurons
     * to the target neuron whose input is being calculated.
     *
     * @param nodes   The neurons of the previous layer
     * @param weights The weights of the edges which connect
     *                the neurons (from nodes) to the target neuron
     *
     * @return This method returns a combination of the nodes
     *         and weights arrays which a neuron can use as its
     *         input for the threshold function.
     *
     * @throws IllegalArgumentException This method throws an IllegalArgumentException
     *                                  if the nodes and weights arrays passed as
     *                                  parameters don't have the same length.
     */
    private double neuronInputCalculator(double[] nodes, double[] weights) {
        if (nodes.length != weights.length)
            throw new IllegalArgumentException("nodes and weights arrays don't have equal lengths");

        // Currently Dot Product
        double input = 0;

        for (int index = 0; index < nodes.length; index++)
            input += nodes[index] * weights[index];

        return input;
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
     * @param input The input to be bounded
     *
     * @return The bounded version of the input
     */
    private double neuronThresholdFunction(double input)
    {
        // Currently f(x) = x
        return input;
    }
}
