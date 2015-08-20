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
package org.ml4j.nn.util;

import org.ml4j.util.NumericFeaturesMatrixCsvDataExtractor;
/**
 * Extracts lines from a csv and converts to pixel vectors.
 * Each line starts with a label (skipped), and then the remaining
 * values are converted to a double[].
 * 
 * @author Michael Lavelle
 *
 */
public class PixelFeaturesMatrixCsvDataExtractor extends NumericFeaturesMatrixCsvDataExtractor {

	@Override
	public double[] createData(String[] csvAttributes) {
		double[] rawData = super.createData(csvAttributes);
		double[] pixelActivationData = new double[rawData.length -1];
		for (int i = 0; i < pixelActivationData.length; i++) {
			pixelActivationData[i] = rawData[i + 1] == 0 ? 0 : 1;
		}
		return pixelActivationData;

	}

}
