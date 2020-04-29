package com.example.project_a.image_classification;

import android.util.Log;

import java.lang.reflect.Array;
import java.util.Scanner;
import com.example.project_a.image_classification.DistanceCalculation;

public class ProcessData {

    float[][] feature_vectors;
    float[][] normalized_feature_vectors;
    int cntFeature = 0;
    int cntLine = 0;
    float[] mean_vector;
    float[] std_vector;

    float[] input_vector;
    float[] normalized_input_vector;

    float[] distanceCalculation_euclideanDistance;
    float[] distanceCalculation_intersectionDistance;
    float[] distanceCalculation_chiSquareDistance;
    float[] distanceCalculation_correlationDistance;

    float average_euclidean_distance_vector;
    float average_intersectionDistance_vector;
    float average_chiSquareDistance_vector;
    float average_correlationDistance_vector;

    float total_average_distance_vector;

    public ProcessData(String sData, float[] input_vector){
        this.input_vector = input_vector;

        //Preprocess for raw feature vectors
        createFeatureVectors(sData);
        calculateMeanVector(feature_vectors);
        calculateSDVector(feature_vectors, mean_vector);

        //Get normalized feature vectors, normalized input vector
        normalizedData(feature_vectors, mean_vector, std_vector);
        normalize_input_feature(this.input_vector, mean_vector, std_vector);
        calculate_Histogram_Distance(normalized_input_vector, normalized_feature_vectors);
        calculate_Average_Histogram_Distance(distanceCalculation_euclideanDistance, distanceCalculation_intersectionDistance, distanceCalculation_chiSquareDistance, distanceCalculation_correlationDistance);
    }
    
    //the output is file with new data that normalizated
    private void createFeatureVectors(String sData) {
        String[] a;
        String[] b;
        Scanner scanner;

        //Counting the number of features and lines
        scanner = new Scanner(sData);
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            a = line.split("n");
            cntLine = a.length;
            b = a[0].split("\\s");
            cntFeature = b.length;
        }
        scanner.close();

        //create a array that contain every feature (Un-normalized data)
        feature_vectors = new float[cntLine][cntFeature];
        scanner = new Scanner(sData);
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String[] lines;
            a = line.split("n");
            //process each line
            for (int i = 0; i < a.length; i++) {
                lines = a[i].split("\\s");
                //process each feature
                for (int j = 0; j < lines.length; j++) {
                    feature_vectors[i][j] = Float.parseFloat(lines[j]);
                }
            }
        }
        scanner.close();
        //Log.d("JUG", String.valueOf(feature_vectors));
    }
    
    private void calculateSDVector(float[][] feature_vectors, float[] mean_vector)
    {
        int row = feature_vectors.length;
        int col = feature_vectors[0].length;
        std_vector = new float[col];
        float[] temp_features = new float[col];
        for(int i = 0; i<col; i++){temp_features[i]=0.0f;}

        for(int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                temp_features[j] += Math.pow(feature_vectors[i][j] - mean_vector[j],2);
            }
        }

        for(int i = 0; i<col; i++){temp_features[i] = (float)Math.sqrt(temp_features[i]/=(row-1));}
        this.std_vector = temp_features;
    }
    
    private void calculateMeanVector(float[][] feature_vectors)
    {
        int row = feature_vectors.length;
        int col = feature_vectors[0].length;
        mean_vector = new float[col];
        float[] sum_features = new float[col];
        for(int i = 0; i<col; i++){sum_features[i]=0.0f;}

        for(int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                sum_features[j] += feature_vectors[i][j];
            }
        }

        for(int i = 0; i<col; i++){sum_features[i]/=row;}
        this.mean_vector = sum_features;
    }

    private void normalizedData(float[][] feature_vectors, float[] mean_vector, float[] std_vector){
        int row = feature_vectors.length;
        int col = feature_vectors[0].length;
        float[][] temp_vectors = new float[row][col];

        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                if(std_vector[j] !=0) {
                    temp_vectors[i][j] = (feature_vectors[i][j] - mean_vector[j]) / std_vector[j];
                }else{
                    temp_vectors[i][j] = (feature_vectors[i][j] - mean_vector[j]);
                }

            }
        }
        this.normalized_feature_vectors = temp_vectors;
    }

    private void normalize_input_feature(float[] input_vector, float[] mean_vector, float[] std_vector){
        float[] temp_vector = new float[input_vector.length];
        for (int i=0; i<input_vector.length; i++){
            if (std_vector[i]!=0) {
                temp_vector[i] = (input_vector[i] - mean_vector[i]) / std_vector[i];
            }else{
                temp_vector[i] = (input_vector[i] - mean_vector[i]);
            }
        }
        this.normalized_input_vector = temp_vector;
    }

    private void calculate_Histogram_Distance(float[] normalized_input_vector, float[][] normalized_feature_vectors){
        //Generate Distance Matrix
        int row = normalized_feature_vectors.length;
        int col = normalized_feature_vectors[0].length;
        float[][] temp_matrix = new float[row][col];
        distanceCalculation_euclideanDistance = new float[row];
        distanceCalculation_intersectionDistance = new float[row];
        distanceCalculation_chiSquareDistance = new float[row];
        distanceCalculation_correlationDistance = new float[row];

        DistanceCalculation distanceCalculation;
        //Calculate Distance for each row
        for (int i=0;i<row;i++){
            float[] normalized_feature_vectors_row = new float[normalized_feature_vectors[i].length];
            for(int j=0;j<col;j++){
                normalized_feature_vectors_row[j] = normalized_feature_vectors[i][j];
            }
            //Calculate Distance
            distanceCalculation = new DistanceCalculation(normalized_feature_vectors_row, normalized_input_vector);
            distanceCalculation_euclideanDistance[i] = distanceCalculation.euclideanDistance(normalized_feature_vectors_row, normalized_input_vector);
            distanceCalculation_intersectionDistance[i] = distanceCalculation.intersectionDistance(normalized_feature_vectors_row, normalized_input_vector);
            distanceCalculation_chiSquareDistance[i] = distanceCalculation.chiSquareDistance(normalized_feature_vectors_row, normalized_input_vector);
            distanceCalculation_correlationDistance[i] = distanceCalculation.correlationDistance(normalized_feature_vectors_row, normalized_input_vector);
        }
    }

    private void calculate_Average_Histogram_Distance(float[] euclideanDistance, float[] intersectionDistance, float[] chiSquareDistance, float[] correlationDistance){
        int row = euclideanDistance.length;
        float[] sum_values = new float[4];
        for (int i=0; i<row;i++){
            sum_values[0] += euclideanDistance[i];
            sum_values[1] += intersectionDistance[i];
            sum_values[2] += chiSquareDistance[i];
            sum_values[3] += correlationDistance[i];
        }
        average_euclidean_distance_vector = sum_values[0]/row;
        average_intersectionDistance_vector = sum_values[1]/row;
        average_chiSquareDistance_vector = sum_values[2]/row;
        average_correlationDistance_vector = sum_values[3]/row;
        total_average_distance_vector = average_euclidean_distance_vector + average_intersectionDistance_vector + average_chiSquareDistance_vector + average_correlationDistance_vector;
    }
}
