
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

import org.ml4j.algorithms.FeaturesMapper;

/**
 * Maps BufferedImage instances of MNIST digits into MNIST-specific format of double[]
 * 
 * @author Michael Lavelle
 *
 */
public class MnistDigitBufferedImageFeaturesMapper implements FeaturesMapper<BufferedImage> {

	private int width;
	private int height;

	public MnistDigitBufferedImageFeaturesMapper(int width, int height) {
		this.width = width;
		this.height = height;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public int getFeatureCount() {
		return width * height;
	}

	@Override
	public double[] toFeaturesVector(BufferedImage image) {
		
		if (image.getWidth() != 28 || image.getHeight() != 28)
		{
			throw new IllegalArgumentException("Image must be 28 * 28 pixels");
		}

		double[] data = new double[image.getWidth() * image.getHeight()];

		int ind = 0;
		for (int w = 0; w < image.getWidth(); w++) {
			for (int h = 0; h < image.getHeight(); h++) {
				int color = image.getRGB(h, w);

				// extract each color component
				int red = (color >>> 16) & 0xFF;
				int green = (color >>> 8) & 0xFF;
				int blue = (color >>> 0) & 0xFF;

				// calc luminance in range 0.0 to 1.0; using SRGB luminance
				// constants
				float luminance = (red * 0.2126f + green * 0.7152f + blue * 0.0722f) / 255;
				// Take the negative of the image for the data
				data[ind++] = 1 - luminance;
			}
		}

		return data;
	}

}
