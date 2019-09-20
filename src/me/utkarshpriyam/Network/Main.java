package me.utkarshpriyam.Network;

import java.io.File;
import java.util.Arrays;

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
public class Main {
   /**
    * This String constant represents the file path to
    * where the files are stored.
    *
    * Update this file path if the files given are stored
    * in a different context to the one provided here.
    *
    * The . refers to the "root" folder for the java program.
    * (ie The outermost folder of the project)
    * From there, follow the relative file path to the files folder.
    */
   private static final String FILE_DIRECTORY = "./src/me/utkarshpriyam/Network/files/";

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
   public static void main(String[] args) {
      // Read all 3 files
      File weights = new File(FILE_DIRECTORY + "weights.txt");
      File inputs = new File(FILE_DIRECTORY + "inputs.txt");
      File outputs = new File(FILE_DIRECTORY + "outputs.txt");

      // Create a new Perceptron with the specified dimensions
      Perceptron pdp = new Perceptron(new int[] {2,2,1});

      // Read the weights for the network from the file
      pdp.readWeights(weights);

      // Run the network on all the inputs (from the file)
      double[][] calculatedOutputs = pdp.runNetwork(inputs);

      // For now, simply print out the output arrays
      for (double[] calculatedOutput: calculatedOutputs)
         System.out.println(Arrays.toString(calculatedOutput));

      // Train network
      pdp.trainNetwork(inputs,outputs);

      // Run the network on all the inputs (from the file)
      calculatedOutputs = pdp.runNetwork(inputs);

      // For now, simply print out the output arrays
      for (double[] calculatedOutput: calculatedOutputs)
         System.out.println(Arrays.toString(calculatedOutput));
   }
}
