/*
    Get image feature from external storage
    Normalization data (Class - ProcessData)
    Calculate Distance (Parameter one feature)
    Prediction result class
 */

package com.example.project_a.image_classification;

import android.os.Environment;
import android.util.Log;

import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Properties;

public class ImageClassification {
    private final int MAX_MATCHES = 50;
    float total_histogram_distance_Hammer;
    float total_histogram_distance_Scissors;
    float total_histogram_distance_Paper;
    float average_ORB_distance_hammer;
    float average_ORB_distance_scissors;
    float average_ORB_distance_paper;

    public String Determine_Class_histogram;
    public String Determine_Class_keypointsdescriptor;

    public ImageClassification(float[] input_feature, String extractedfeatures_hammer, String extractedfeatures_scissors,
                               String extractedfeatures_paper, MatOfKeyPoint keypointsORB_input_feature, Mat descriptorsORB_input_feature,
                               ArrayList<String> listOf_ProcessedImagesFile, Properties listOfAllImages_Class, ArrayList<MatOfKeyPoint> keypointsORB_all_images,
                               ArrayList<Mat> descriptorsORB_all_images, String PredictionMethod){

        if(PredictionMethod.equals("HISTOGRAM")){
            //Calculate Histogram distance
            get_total_histogram_distance(input_feature, extractedfeatures_hammer, extractedfeatures_scissors, extractedfeatures_paper);
            //Determine Class
            Determine_Class_histogram = determine_class_by_histogram_distance( total_histogram_distance_Hammer, total_histogram_distance_Scissors, total_histogram_distance_Paper);
        }else if(PredictionMethod.equals("KEYPOINTSDESCRIPTOR")){
            //Caluculate ORB matching distance
            float[] sum_distance = get_matching_ORB_descriptors_distance(keypointsORB_input_feature, descriptorsORB_input_feature, listOf_ProcessedImagesFile, keypointsORB_all_images, descriptorsORB_all_images);
            //Filtering distance value with respect to class
            get_ORB_matching_distance(sum_distance, listOf_ProcessedImagesFile, listOfAllImages_Class);
            //Determine Class
            Determine_Class_keypointsdescriptor = determine_class_by_keypoints_descriptor(average_ORB_distance_hammer,average_ORB_distance_scissors, average_ORB_distance_paper);
        }
    }

    private void get_total_histogram_distance(float[] input_feature, String extractedfeatures_hammer, String extractedfeatures_scissors, String extractedfeatures_paper){
        ProcessData processData_hammer = new ProcessData(extractedfeatures_hammer, input_feature);
        ProcessData processData_scissors = new ProcessData(extractedfeatures_scissors, input_feature);
        ProcessData processData_paper = new ProcessData(extractedfeatures_paper, input_feature);
        total_histogram_distance_Hammer = processData_hammer.total_average_distance_vector;
        total_histogram_distance_Scissors = processData_scissors.total_average_distance_vector;
        total_histogram_distance_Paper = processData_paper.total_average_distance_vector;
    }

    private float[] get_matching_ORB_descriptors_distance(MatOfKeyPoint keypointsORB_input_feature, Mat descriptorsORB_input_feature,
                   ArrayList<String> listOf_ProcessedImagesFile, ArrayList<MatOfKeyPoint> keypointsORB_all_images, ArrayList<Mat> descriptorsORB_all_images){

        //Keypoints and Descriptor for input image
        float[] sum_distance = new float[listOf_ProcessedImagesFile.size()];

        for(int i=0; i<listOf_ProcessedImagesFile.size();i++){
            MatOfKeyPoint keypointsORB_match_feature = keypointsORB_all_images.get(i);
            Mat descriptorsORB_match_feature = descriptorsORB_all_images.get(i);
            //Calculate distance of matching keypoints
            sum_distance[i] = matching_ORB_keypoint_descriptor_distance(keypointsORB_input_feature, keypointsORB_match_feature, descriptorsORB_input_feature, descriptorsORB_match_feature);
        }
        return sum_distance;
    }

    private float matching_ORB_keypoint_descriptor_distance(MatOfKeyPoint keypoints1, MatOfKeyPoint keypoints2, Mat descriptors1, Mat descriptors2){
        //MatOfKeyPoint keypoints1, keypoints2;
        //Mat descriptors1, descriptors2;
        //FeatureDetector detector = null;
        //detector = FeatureDetector.create(FeatureDetector.ORB);
        float sum_distance = 0.0f;

        MatOfDMatch matches = new MatOfDMatch();
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        //Matching 2 descriptor
        descriptorMatcher.match(descriptors1, descriptors2, matches);

        Collections.sort(matches.toList(), new Comparator<DMatch>() {
            @Override
            public int compare(DMatch o1, DMatch o2) {
            if(o1.distance<o2.distance)
                return -1;
            if(o1.distance>o2.distance)
                return 1;
            return 0;
            }
        });

        //Get Max Match (50 first lowest distance value)
        List<DMatch> listOfDMatch = matches.toList();
        if(listOfDMatch.size()>MAX_MATCHES){
            matches.fromList(listOfDMatch.subList(0,MAX_MATCHES));
        }
        //Store distance values
        float []distance_orb = new float[matches.toList().size()];
        sum_distance = 0.0f;
        for (int i=0; i < matches.toList().size(); i++){
            //distance[i] = listOfDMatch.get(i).distance;
            distance_orb[i] = matches.toList().get(i).distance;
            sum_distance = sum_distance + distance_orb[i];
        }
        return sum_distance;
    }

    private void get_ORB_matching_distance(float[] sum_distance, ArrayList<String> listOf_ProcessedImagesFile, Properties listOfAllImages_Class){
        //Filtering file name with respect to class name
        ArrayList<String> filter_file_Hammer = filtering_file_class(listOfAllImages_Class, "HAMMER");
        ArrayList<String> filter_file_Scissors = filtering_file_class(listOfAllImages_Class, "SCISSORS");
        ArrayList<String> filter_file_Paper = filtering_file_class(listOfAllImages_Class, "PAPER");

        //Filtering distance with respect to file name
        float[] filter_distance_values_Hammer = filtering_distance_file(sum_distance, listOf_ProcessedImagesFile, filter_file_Hammer);
        float[] filter_distance_values_Scissors = filtering_distance_file(sum_distance, listOf_ProcessedImagesFile, filter_file_Scissors);
        float[] filter_distance_values_Paper = filtering_distance_file(sum_distance, listOf_ProcessedImagesFile, filter_file_Paper);

        //Calculate average distance for each class
        average_ORB_distance_hammer = 0.0f;
        for (int i=0; i<filter_distance_values_Hammer.length; i++){
            average_ORB_distance_hammer += filter_distance_values_Hammer[i];
        }
        average_ORB_distance_hammer/=filter_distance_values_Hammer.length;

        average_ORB_distance_scissors = 0.0f;
        for (int i=0; i<filter_distance_values_Scissors.length; i++){
            average_ORB_distance_scissors += filter_distance_values_Scissors[i];
        }
        average_ORB_distance_scissors/=filter_distance_values_Scissors.length;

        average_ORB_distance_paper = 0.0f;
        for (int i=0; i< filter_distance_values_Paper.length; i++){
            average_ORB_distance_paper += filter_distance_values_Paper[i];
        }
        average_ORB_distance_paper/=filter_distance_values_Paper.length;
    }

    private ArrayList<String> filtering_file_class(Properties listOfAllImages_Class, String FilteringClass){
        //Filtering distance values
        //Filter if there is the same property
        ArrayList<String> filter_file_class = new ArrayList<String>();
        Enumeration e = listOfAllImages_Class.keys();
        while(e.hasMoreElements()){
            String str = (String)e.nextElement();
            if(listOfAllImages_Class.get(str).equals(FilteringClass)){
                filter_file_class.add(str);
            }
        }
        return filter_file_class;
    }

    private float[] filtering_distance_file(float[] sum_distance, ArrayList<String> listOf_ProcessedImagesFile, ArrayList<String> filter_file_class){
        float[] filter_distance_values_class;
        List<Float> filter_distance_values = new ArrayList<Float>();
        int idx_match_file;
        int k = 0;
        for(int i = 0; i<filter_file_class.size(); i++){
            String filter_file_name = filter_file_class.get(i);
            for(int j=0; j<listOf_ProcessedImagesFile.size(); j++){
                List<String> list = new ArrayList<String>(Arrays.asList(listOf_ProcessedImagesFile.get(j).split("/")));
                if (list.get(list.size()-1).equals(filter_file_name)){
                    filter_distance_values.add(sum_distance[j]);
                }
            }
        }
        filter_distance_values_class = new float[filter_distance_values.size()];
        for (int i=0;i<filter_distance_values.size();i++){
            filter_distance_values_class[i] = filter_distance_values.get(i);
        }
        return filter_distance_values_class;
    }

    private String determine_class_by_histogram_distance(float total_histogram_distance_Hammer, float total_histogram_distance_Scissors, float total_histogram_distance_Paper){

        float Min_distance = 99999.0f;
        String class_by_histogram = "";

        if (total_histogram_distance_Hammer < Min_distance){
            Min_distance = total_histogram_distance_Hammer;
            class_by_histogram = "HAMMER";
        }
        if (total_histogram_distance_Scissors < Min_distance){
            Min_distance = total_histogram_distance_Scissors;
            class_by_histogram = "SCISSORS";
        }
        if (total_histogram_distance_Paper < Min_distance){
            Min_distance = total_histogram_distance_Paper;
            class_by_histogram = "PAPER";
        }
        return class_by_histogram;
    }

    private String determine_class_by_keypoints_descriptor(float average_ORB_distance_hammer,float average_ORB_distance_scissors, float average_ORB_distance_paper){
        float Min_distance = 99999.0f;
        String class_by_decriptor = "";

        if(average_ORB_distance_hammer < Min_distance){
            Min_distance = average_ORB_distance_hammer;
            class_by_decriptor = "HAMMER";
        }
        if(average_ORB_distance_scissors < Min_distance){
            Min_distance = average_ORB_distance_scissors;
            class_by_decriptor = "SCISSORS";
        }
        if(average_ORB_distance_paper < Min_distance){
            Min_distance = average_ORB_distance_paper;
            class_by_decriptor = "PAPER";
        }
        return class_by_decriptor;
    }
}
