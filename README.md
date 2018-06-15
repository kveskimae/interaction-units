Higher order networks with interaction units
==================================

Experimental implementations of multilayer neural network using `numpy` library and some additional modules, based on [Digit Classifier by Desai](https://github.com/kdexd/digit-classifier) , and adding interaction layers.

## All pairwise interactions

In this configuration, interaction terms are formed by making all possible pairwise multiplications.

![Pairwise interactions](/img/all_pairwise.png)

In actual implementation, the inputs to pairwise interaction nodes are not weighted. Also, bias and activation are skipped for these interaction nodes. So the input is effectively transformed into new feature space.

Network layout that was tested for all pairwise interactions is following

![Network layout for all pairwise interactions](/img/layout_all_pairwise.png)

## Correlations

The most most common way to combine inputs for interaction is via weighted pairwise correlations.

![Pairwise correlations](/img/correlations.png)

Formula for forward pass for activation of j-th neuron in the l-th layer is
* for weight layer 

![Forward pass for weight layer](/img/equation_forward_pass_weight_layer.png)

* VS now for the new interaction layer 

![Forward pass for interaction layer](/img/equation_forward_pass_interaction_layer.png)

Two network layouts were tested for correlations.

Configuration 1 stacks correlations layer after the first hidden layer:

![Configuration 1](/img/correlations_conf1_layout.png)

Configuration 2 stacks correlations layer right after the input layer:

![Configuration 1](/img/correlations_conf2_layout.png)

## Code files

Work is in Jupyter notebooks in Python 2.7 :
* *correlations_conf1.ipynb* contains a correlations layer (with activation and bias) as in configuration 1;
* *correlations_conf2.ipynb* contains a correlations layer (with activation and bias) as in configuration 2;
* *reference.ipynb* contains the original reference network with the same layout as configuration 1, but with ordinary sigmoid neurons in place of interaction units;
* *all_pairwise.ipynb* stacks a layer of all pairwise interactions (without activation or bias) after the input layer.

## Results

In evaluating networks, training went through 20 epochs.

![Test accuracies](/img/test_accuracy.png) 

![Validation accuracies](/img/Validation.png)