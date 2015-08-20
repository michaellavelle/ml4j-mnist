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
import org.ml4j.nn.FeedForwardLayer;
import org.ml4j.nn.FeedForwardNeuralNetwork;
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
 * Trains a Feed Forward Neural Network to recognise MNIST digits
 * 
 * @author Michael Lavelle
 *
 */
public class DigitClassificationFNNTrainer {

	public static void main(String[] args) throws IOException, InterruptedException {

	
		SerializationHelper helper = new SerializationHelper(DigitClassificationFNNTrainer.class.getClassLoader(),
				"org/ml4j/mnist");
		
		DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
				DigitClassificationFNNTrainer.class.getClassLoader());

		// Load Mnist data into double[][] matrices. Load the first x
		// records to be used as a training set, and the last 10000 records for the test set.

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

		// By default assume JBlas is available (the case on Macs) - set to
		// false if JBlas not available to use slower JAMA
		boolean jBlasAvailable = true;

		// First layer takes inputs from 28 * 28 input Neurons, and activates 500 hidden Neurons
		FeedForwardLayer firstLayer = new FeedForwardLayer(28 * 28,500, new SigmoidActivationFunction(),true);
		
		// Second layer takes inputs from 500 Neurons and activates 500 hidden Neurons
		FeedForwardLayer secondLayer = new FeedForwardLayer(500,500, new SigmoidActivationFunction(),true);
		
		// Third layer takes inputs from 500 Neurons and activates 2000 hidden Neurons
		FeedForwardLayer thirdLayer = new FeedForwardLayer(500,2000, new SigmoidActivationFunction(),true);
		
		// Forth layer takes inputs from 200 Neurons and activates 10 output Neurons in a 10-way softmax
		FeedForwardLayer forthLayer = new FeedForwardLayer(2000,10, new SoftmaxActivationFunction(),true);

		FeedForwardNeuralNetwork neuralNetwork = new FeedForwardNeuralNetwork(firstLayer,secondLayer,thirdLayer,forthLayer);

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

		System.out.println("\nTraining...\n");
		NeuralNetworkHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix, trainingLabelsMatrix,
				context);

		// Training Set accuracy
		System.out.println("Accuracy on training set:" + hyp1.getAccuracy(trainingDataMatrix, trainingLabelsMatrix));

		// Test Set accuracy
		System.out.println("Accuracy on test set:" + hyp1.getAccuracy(testSetDataMatrix, testSetLabelsMatrix));

		// Serialize the hypothesis function
		String serializedHypothesisFunctionName = "workingFFNHypothesisFunction";	
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
