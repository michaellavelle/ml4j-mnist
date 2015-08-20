/*
 * Copyright 2015 the original author or authors.
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
package org.ml4j.mnist.demos;

import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.mnist.service.DigitClassificationService;
import org.ml4j.mnist.service.NeuralNetworkDigitClassificationService;
import org.ml4j.mnist.training.DigitClassificationCNNTrainer;
import org.ml4j.nn.algorithms.NeuralNetworkHypothesisFunction;
import org.ml4j.nn.util.MnistUtils;
import org.ml4j.nn.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.ml4j.util.SerializationHelper;
/**
 * Classifies MNIST digits (from the raw data) using pre-learned Neural Network hypothesis function
 * 
 * A serialized hypothesis function is loaded from the classpath by name
 * 
 * @author Michael Lavelle
 *
 */
public class DigitImageRawDataClassifierDemo {

	public static void main(String[] args) throws InterruptedException
	{
		// Assumed that GPU Cuda optimisation is disabled by default - enable for faster performance if CUDA available
		boolean cudaAvailable = false;
		
		// Assumed that JBlas is available by default (the case on Macbooks) - disable to fall back to JAMA matrix strategy (slower)
		boolean jBlasAvailable = true;
		
		String serializedHypothesisFunctionName = "19_08_2015_CNN_1";
	
		// Whether to display the images as we are predicting them
		boolean displayImages = false;
		
		// Load cross validation set data and labels from 10000 records towards end of text file that haven't been seen before during training
		DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
				DigitClassificationCNNTrainer.class.getClassLoader());

		double[][] crossValidationDataMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 32005, 42005);

		double[][] crossValidationLabelsMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 32005, 42005);

		NeuralNetworkHypothesisFunction preLearnedHypothesisFunction
		 =getPreTrainedHypothesisFunction(serializedHypothesisFunctionName);
		
		// Output the underlying Neural Network configuration of the pre-trained model
		System.out.println("Neural Network Configuration...\n");
		System.out.println(preLearnedHypothesisFunction.getNeuralNetwork());
		
		DigitClassificationService digitClassificationService
		 = new NeuralNetworkDigitClassificationService(preLearnedHypothesisFunction,cudaAvailable,jBlasAvailable);
	
		System.out.println("Computing accuracy on entire cross validation set...\n");

		System.out.println("Accuracy on cross validation set:" + preLearnedHypothesisFunction.getAccuracy(crossValidationDataMatrix, crossValidationLabelsMatrix));

		Thread.sleep(2000);

		
		// For demo purposes, output the predicted and actual labels and optionally display images
		System.out.println("\nOutputting 100 predictions\n");
		ImageDisplay<Long> display = null;
		if (displayImages)
		{
			display = new ImageDisplay<Long>(280, 280);
		}
		for (int i = 0; i < 100; i++) {

			// For each element in our test set, obtain the predicted and actual
			// classification
			int predicted = digitClassificationService.getPredictedDigitClassification(crossValidationDataMatrix[i]);
			int actual = NeuralNetworkDigitClassificationService.getArgMaxIndex(crossValidationLabelsMatrix[i]);

			// Output prediction
			System.out.println("Predicted:" + predicted + ",Actual:" + actual);
			
			// Display the actual input image
			if (displayImages)
			{
				MnistUtils.draw(crossValidationDataMatrix[i], display);
				Thread.sleep(1000);
			}

		}
		System.exit(0);

		
	}
	
	private static NeuralNetworkHypothesisFunction getPreTrainedHypothesisFunction(String serializedHypothesisFunctionName)
	{
		if (serializedHypothesisFunctionName.isEmpty())
		{
			throw new RuntimeException("Serialized hypothesis function name is unspecified");
		}
		try
		{
			// De-serialize pre-trained neural network hypothesis function
			SerializationHelper serializationhelper = new SerializationHelper(DigitImageClassifierDemo.class.getClassLoader(),"org/ml4j/mnist");
			
			NeuralNetworkHypothesisFunction preLearnedHypothesisFunction
			 = serializationhelper.deserialize(NeuralNetworkHypothesisFunction.class, serializedHypothesisFunctionName);
			
			return preLearnedHypothesisFunction;
		}
		catch (Exception e)
		{
			// Need to raise correct exceptions in SerializationHelper - TODO, but
			// now, for demo purposes, throw RuntimeException explaining unable to load.
			throw new RuntimeException("Unable to load hypothesis function:" + serializedHypothesisFunctionName);
		}
		
	}
	
}
