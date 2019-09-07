package me.utkarshpriyam.Network_v1;

import java.io.File;
import java.util.Arrays;

/**
 * This is the main class which will run the network
 *
 * @author Utkarsh Priyam
 * @version 9/4/19
 */
public class Main {

    public static void main(String[] args) {
        String baseDir = "/Users/up/Desktop/CS/Projects/ATCS/" +
                "Neural Networks/Network v1 Java/src/me/utkarshpriyam/Network_v1/files/";
        File weights = new File(baseDir + "weights.txt"), inputs = new File(baseDir + "inputs.txt");
        Perceptron pdp = new Perceptron(new int[] {2,2,1});
        pdp.setWeights(weights);
        double[][] outputs = pdp.runNetwork(inputs);
        for (double[] output: outputs)
            System.out.println(Arrays.toString(output));
    }
}
