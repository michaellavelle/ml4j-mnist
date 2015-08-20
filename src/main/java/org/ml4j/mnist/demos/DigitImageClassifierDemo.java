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

import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.ml4j.mnist.service.DigitClassificationService;
import org.ml4j.mnist.service.NeuralNetworkDigitClassificationService;
import org.ml4j.nn.algorithms.NeuralNetworkHypothesisFunction;
import org.ml4j.nn.util.JpegFileFilter;
import org.ml4j.util.SerializationHelper;
/**
 * Classifies images of MNIST digits using pre-learned Neural Network hypothesis function
 *  * 
 * A serialized hypothesis function is loaded from the classpath by name
 * 
 * @author Michael Lavelle
 *
 */
public class DigitImageClassifierDemo {

	public static void main(String[] args) throws InterruptedException, IOException
	{
		// Assumed that GPU Cuda optimisation is disabled by default - enable for faster performance if CUDA available
		boolean cudaAvailable = false;
		
		// Assumed that JBlas is available by default (this is case on MacBooks) - disable to fall back to JAMA matrix strategy (slower)
		boolean jBlasAvailable = true;
	
		// Load pre-trained Neural network hypothesis function
		String serializedHypothesisFunctionName = "19_08_2015_CNN_1";
		
		NeuralNetworkHypothesisFunction preLearnedHypothesisFunction
		 =getPreTrainedHypothesisFunction(serializedHypothesisFunctionName);
		
		// Output the underlying Neural Network configuration of the pre-trained model
		System.out.println("Neural Network Configuration...\n");
		System.out.println(preLearnedHypothesisFunction.getNeuralNetwork());
		
		
		// Create our digit classification service, configuring with the pre-trained NN hyp function
		DigitClassificationService digitClassificationService
		 = new NeuralNetworkDigitClassificationService(preLearnedHypothesisFunction,cudaAvailable,jBlasAvailable);
	
		
		// Load images from src/main/resource/test_images, and obtain predictions
		File imagesDirectory = new File(DigitImageClassifierDemo.class.getClassLoader().getResource("test_images").getFile());		
		JpegFileFilter jpegFileFilter = new JpegFileFilter();
		
		for (File imageFile : imagesDirectory.listFiles(jpegFileFilter)) {

			int prediction = digitClassificationService.getPredictedDigitClassification(ImageIO.read(imageFile));		
			System.out.println(imageFile.getAbsolutePath() + ": Predicted " +  prediction);
		}
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
