File Configuration File Layout:
-------------------------------
DESCRIPTION: This file gives a listing of all the individual control files needed for the network to run

Line 1: Inputs File
  -- Description in Inputs File section
Line 2: Outputs File
  -- Description in Outputs File section
Line 3: Weights File
  -- Description in Weights File section
Line 4: Weight Dump
  -- Description in Weight Dump section
Line 5: Output Dump
  -- Description in Output Dump section
Line 6: Other Dump
  -- Description in Other Dump section

Network Configuration File Layout:
----------------------------------
DESCRIPTION: This file gives a listing of all the parameters necessary for the network to run properly

Line 1: Network Structural Configuration (A-B-...)
  -- A list of hyphen-separated integers that give the dimensions of the network
Line 2: Whether to Run, Train, or Test the network
  -- All lowercase letters spelling "run", "train", or "test" (without the quotation marks)
Line 3: Number of Test Cases to Execute Over
  -- A single integer that tells the network how many test cases there are
Line 4: Starting LAMBDA value, LAMBDA change value, Minimum LAMBDA cap, Maximum LAMBDA cap
  -- A list of 4 space-separated doubles (in this order)
  -- Results guaranteed only if: Minimum LAMBDA cap < Starting LAMBDA value < Maximum LAMBDA cap
Line 5: The Minimum Error Under Which the Network Can Declare Success
  -- Set this to 0 if you want to get rid of the minimum error success exit case
Line 6: The Maximum Number of Iterations Before the Network Stops Training
  -- This must be an integer or it will be defaulted to 100000 (100k = one hundred thousand) iterations
Line 7: Random Weights Generation Bounds: Low, High
  -- A list of 2 space separated doubles
  -- The random weights will generate in the range [Low, High)
  -- Results only guaranteed if: Low < High
Line 8: Whether to Generate the Underlying Arrays as RAGGED Arrays or BOX Arrays
  -- "true" = RAGGED Array
  -- "false" = BOX Array
  -- Any Other String = BOX Array
  -- Right now there is only support for BOX Arrays

Inputs File Layout:
-------------------
DESCRIPTION: This file gives all the input test cases for the network (used in all 3 run modes)

Line 1: A Single Integer (N) Telling how Many Test Cases There Are
Line 2: Test Case 1
        .
        .
        .
Ln N+1: Test Case N
  -- All the Test Case lines must be lists of integers
    -- The number of inputs in each case must be equal to
       the number of input nodes in the network

Outputs File Layout:
--------------------
DESCRIPTION: This file gives all the output test cases for the network (used "train" and "test" run modes)

Line 1: A Single Integer (N) Telling how Many Test Cases There Are
  -- This number MUST match the number in the Inputs File
Line 2: Test Case 1
        .
        .
        .
Ln N+1: Test Case N
  -- All the Test Case lines must be lists of integers
    -- The number of outputs in each case must be equal to
       the number of output nodes in the network

Weights File Layout:
--------------------
DESCRIPTION: This file holds the starting weights for the network
DESCRIPTION: Any omitted/malformed weights will be randomized

Weight Dump File Layout:
------------------------
DESCRIPTION: This is the file where the finishing weights are dumped (useful after training)
DESCRIPTION: The dumped weights will be in a format where this file can be fed back into the network
NOTE: Any data previously present in the file will be lost

Output Dump File Layout:
------------------------
DESCRIPTION: This is the file where the finishing outputs are dumped (useful for finely combing the outputs)
NOTE: Any data previously present in the file will be lost

Other Dump File Layout:
-----------------------
ULTRA-IMPORTANT NOTE: This feature is currently not implemented
  -- Most of these values will be printed out to System.out anyways
DESCRIPTION: This is the file where other important values such as LAMBDA hyperparameters will be dumped
NOTE: Any data previously present in the file will be lost