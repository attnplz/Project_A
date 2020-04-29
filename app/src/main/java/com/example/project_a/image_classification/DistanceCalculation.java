/*
     https://math.stackexchange.com/questions/2383969/properly-comparing-two-histograms
     http://man.hubwiz.com/docset/OpenCV.docset/Contents/Resources/Documents/d8/dc8/tutorial_histogram_comparison.html
*/

package com.example.project_a.image_classification;

import org.opencv.core.Mat;

public class DistanceCalculation {
    private float[] vector1;
    private float[] vector2;
    private double euclidean_distance_value;


    public DistanceCalculation(float[] vector1, float[] vector2){
        this.vector1 = vector1;
        this.vector2 = vector2;
    }

    public float euclideanDistance(float[] vector1, float[] vector2){
        float distance = 0.0f;
        for(int i = 0; i<vector1.length; i++){
            distance += Math.pow(vector1[i] - vector2[i],2);
        }
        distance = Math.abs(distance);
        return distance;
    }

    public float intersectionDistance(float[] vector1, float[] vector2) {
        float distance = 0.0f;
        for(int i = 0; i<vector1.length; i++){
            distance += Math.min(vector1[i], vector2[i]);
        }
        return distance;
    }

    public float chiSquareDistance(float[] vector1, float[] vector2){
        float distance = 0.0f;
        float numerator;

        for(int i = 0; i<vector1.length; i++){
            numerator = (float)Math.pow(vector1[i] - vector2[i],2);
            if (vector1[i]!=0) {
                distance += numerator / vector1[i];
            }else{
                distance += numerator;
            }
        }
        return distance;
    }

    public float correlationDistance(float[] vector1, float[] vector2){
        float mean_vector1, mean_vector2;
        float numerator = 0.0f, temp1 = 0.0f, temp2 = 0.0f;
        float distance = 0.0f;

        mean_vector1 = meanValue(vector1);
        mean_vector2 = meanValue(vector2);

        for(int i = 0; i<vector1.length; i++){
            numerator += (vector1[i] - mean_vector1) * (vector2[i] - mean_vector2);
            temp1 += (float)Math.pow(vector1[i] - mean_vector1,2);
            temp2 += (float)Math.pow(vector2[i] - mean_vector2,2);
        }
        if (temp1==0 || temp2==0){
            distance = 0;
        }else {
            distance = numerator / (float) Math.sqrt(temp1 * temp2);
        }
        return distance;
    }

    private float meanValue(float[] vector){
        float mean_value = 0.0f;
        float sum_value = 0.0f;
        for(int i=0; i<vector.length; i++) {
            sum_value += vector[i];
        }
        mean_value = sum_value/vector.length;
        return mean_value;
    }


}
