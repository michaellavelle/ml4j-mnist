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
package org.ml4j.mnist.service;

import java.awt.image.BufferedImage;

/**
 * Digit classification services
 * 
 * @author Michael Lavelle
 *
 */
public interface DigitClassificationService {

	/**
	 * 
	 * @param mnistData a 28 * 28 image represented as a double[]
	 * @return The predicted digit
	 */
	public int getPredictedDigitClassification(double[] mnistData);
	
	/**
	 * 
	 * @param mnistData array of 28 * 28 images, each represented as a double[]
	 * @return The predicted digit
	 */
	public int[] getPredictedDigitClassifications(double[][] mnistData);
	
	/**
	 * 
	 * @param image The image to classify ( must be 28 * 28)
	 * @return The predicted digit
	 */
	public int getPredictedDigitClassification(BufferedImage image);
	
	/**
	 * 
	 * @param testSetData A mnistData array of 28 * 28 images, each represented as a double[]
	 * @param testSetLabels An array of labels, each represented as a double[] with the index of the '1' element identifying the digit 
	 * @return
	 */
	public double getAccuracy(double[][] testSetData,double[][] testSetLabels);

}
