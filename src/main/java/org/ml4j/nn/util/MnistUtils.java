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

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.targets.ImageDisplay;

/**
 * Utility class for Mnist
 * 
 * @author Michael Lavelle
 *
 */
public class MnistUtils {

	public static void draw(double[] data, ImageDisplay<Long> imageDisplay) {
		double[] pixelData = new double[data.length];

		// Rearrange ordering of pixels for display purposes ( default image is
		// flipped),
		// and scale to greyscale range, and reverse zeros and ones to match
		// reversed
		// input data
		int in = 0;
		for (int r1 = 0; r1 < 28; r1++) {
			for (int c = 0; c < 28; c++) {
				int o = (r1) * 28 + (c);
				double originalValue = data[o] * 255;
				double reversedValue = 255 - originalValue;
				pixelData[in++] = reversedValue;
			}
		}

		BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);

		WritableRaster r = img.getRaster();
		byte[] equiv = new byte[pixelData.length];
		for (int i = 0; i < equiv.length; i++) {
			equiv[i] = new Double(pixelData[i]).byteValue();
		}
		r.setDataElements(0, 0, 28, 28, equiv);

		// Resize and display the image
		BufferedImage resized = new BufferedImage(280, 280, img.getType());
		Graphics2D g = resized.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g.drawImage(img, 0, 0, 280, 280, 0, 0, 28, 28, null);
		g.dispose();
		imageDisplay.onFrameUpdate(new SerializableBufferedImageAdapter(resized), 1000l);
	}
	

}
