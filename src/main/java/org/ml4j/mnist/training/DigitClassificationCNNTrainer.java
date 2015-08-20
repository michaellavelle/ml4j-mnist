/*
` * Copyright 2015 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.mnist.training;

import java.io.IOException;

import org.ml4j.ConvertToCudaMatrixOptimisationStrategy;
import org.ml4j.CudaForMMulStrategy;
import org.ml4j.DefaultMatrixAdapterStrategy;
import org.ml4j.DoubleMatrixConfig;
import org.ml4j.NoOpMatrixOptimisationStrategy;
import org.ml4j.jblas.NoJblasPresentMatrixAdapterStrategy;
import org.ml4j.nn.ConvolutionalLayer;
import org.ml4j.nn.FeedForwardLayer;
import org.ml4j.nn.FeedForwardNeuralNetwork;
import org.ml4j.nn.MaxPoolingLayer;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithm;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.NeuralNetworkHypothesisFunction;
import org.ml4j.nn.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.ml4j.util.SerializationHelper;

/**
 * Trains a Convolutional Neural Network to recognise MNIST digits
 * 
 * @author Michael Lavelle
 *
 */
public class DigitClassificationCNNTrainer {

	public static void main(String[] args) throws IOException, InterruptedException {

		// Load Mnist data into double[][] matrices. Load the first x
		// records to be used as a training set, and the last 10000 records for the test set.

		
		DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
				DigitClassificationCNNTrainer.class.getClassLoader());

		double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 1, 1001);
		double[][] testSetDataMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 32005, 42005);
		double[][] trainingLabelsMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 1, 1001);
		double[][] testSetLabelsMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 32005, 42005);
	
		
		// By default, assume CUDA is not available - set to true to use GPU
		// matrix-matrix multiplies
		boolean cudaAvailable = false;

		// By default assume JBlas is available (the case on Macbooks) - set to
		// false if JBlas not available to use slower JAMA
		boolean jBlasAvailable = true;

		// Configure a Neural Network, with configurable hidden neuron topology,
		// and classification output neurons corresponding to the 10 numbers to
		// be predicted.

		// First layer is a Convolutional layer, taking images of 784 (28 * 28)
		// pixels, and applying 6 convolutional filters of
		// size (9 * 9)to generate 6 feature maps of size ( 20 * 20)

		FeedForwardLayer firstLayer = new ConvolutionalLayer(784, 6 * 20 * 20, new SigmoidActivationFunction(), true,
				6, 1);

		// Second layer is a max pooling layer which subsamples the 20 * 20
		// feature maps to

		FeedForwardLayer secondLayer = new MaxPoolingLayer(6 * 20 * 20, 6 * 10 * 10, 6);

		// Third layer is another convolutional layer, taking 6 * (10 * 10)
		// feature maps and applying 16 filters of 6 * 6 to generate 16 feature
		// maps of 5 * 5

		FeedForwardLayer thirdLayer = new ConvolutionalLayer(6 * 10 * 10, 16 * 5 * 5, new SigmoidActivationFunction(),
				true, 16, 6);

		// Fourth layer is a fully connected layer, taking 16 * ( 5 * 5 )
		// feature maps and connecting to 100 hidden neurons

		FeedForwardLayer forthLayer = new FeedForwardLayer(16 * 5 * 5, 100, new SigmoidActivationFunction(), true);

		// Fifth layer is a fully connected layer, taking 100 hidden neurons and
		// connecting to 10 softmax output neurons
		// representing the 10 digit classes.

		FeedForwardLayer fifthLayer = new FeedForwardLayer(100, 10, new SoftmaxActivationFunction(), true);

		FeedForwardNeuralNetwork neuralNetwork = new FeedForwardNeuralNetwork(firstLayer, secondLayer, thirdLayer,
				forthLayer, fifthLayer);

		System.out.println(neuralNetwork);
		
		// Make JBlas/Cuda optimisations
		makeJblasAndCudaOptimisations(neuralNetwork,cudaAvailable,jBlasAvailable);
		
		// Create algorithm
		NeuralNetworkAlgorithm alg = new NeuralNetworkAlgorithm(neuralNetwork);

		// Create training context
		int iterations = 100;
		NeuralNetworkAlgorithmTrainingContext context = new NeuralNetworkAlgorithmTrainingContext(iterations);

		// Choose amount of regularisation ( configure this to reduce generalisation error)
		double regularizationLambda = 0d;

		context.setRegularizationLambda(regularizationLambda);

		// Generate hypothesis function from algorithm

		System.out.println("\nTraining...");
		NeuralNetworkHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix, trainingLabelsMatrix,
				context);

		// Training Set accuracy
		System.out.println("Accuracy on training set:" + hyp1.getAccuracy(trainingDataMatrix, trainingLabelsMatrix));

		// Test Set accuracy
		System.out.println("Accuracy on test set:" + hyp1.getAccuracy(testSetDataMatrix, testSetLabelsMatrix));

		// Serialize the hypothesis function
		String serializedHypothesisFunctionName = "workingCNNHypothesisFunction";	
		SerializationHelper helper = new SerializationHelper(DigitClassificationCNNTrainer.class.getClassLoader(),
				"org/ml4j/mnist");
		
		helper.serialize(hyp1, serializedHypothesisFunctionName);
	}

	

	private static void makeJblasAndCudaOptimisations(FeedForwardNeuralNetwork neuralNetwork, boolean cudaAvailable,
			boolean jBlasAvailable) {
		if (!cudaAvailable) {
			// The hypothesis functions may have been generated by CUDA matrix
			// strategies, so unconfigure these strategies if we don't want to
			// use CUDA
			if (jBlasAvailable) {
				DoubleMatrixConfig.setDoubleMatrixStrategy(new DefaultMatrixAdapterStrategy());
			} else {
				DoubleMatrixConfig.setDoubleMatrixStrategy(new NoJblasPresentMatrixAdapterStrategy());

			}
			neuralNetwork
					.updateForwardPropagationInputMatrixStrategyForCurrentLayers(new NoOpMatrixOptimisationStrategy());
		} else {
			// If Cuda is available, optimise for GPU matrix-matrix
			// multiplication
			DoubleMatrixConfig.setDoubleMatrixStrategy(new CudaForMMulStrategy());
			neuralNetwork
					.updateForwardPropagationInputMatrixStrategyForCurrentLayers(new ConvertToCudaMatrixOptimisationStrategy());
		}

	}

}
