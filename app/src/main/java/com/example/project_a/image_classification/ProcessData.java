/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.example.project_a.image_classification;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author TongC
 */
public class ProcessData {

    public static void main(String[] args) {
        
        //normalization and return a new file in the project folder
        normalization();
        
        //normalization with input and return an array
        Double[] result = improveData(new Double[] {111.0,112.0,113.0,114.0,115.0});
        for (int i=0; i<result.length; i++) {
            System.out.print(result[i]+" ");
        }
    }
    
    //the output is file with new data that normalizated
    public static void normalization() {
       
        //phase1 virable extract every data that nesscery from file
        int cntFeature =0 ;
        int cntLine =0 ;
        String[] a ;
        
        try {
             //phase1
             //count  a number of features and lines
             File file = new File("dataset.txt");
             Scanner sc = new Scanner(file);
             while (sc.hasNextLine()) {
                    a =  sc.nextLine().split("\\s");
                    cntFeature = a.length;
                    cntLine++;
             } 
              sc.close();
             
              //create a array that contain every feature
              Double[][] feature = new Double[cntFeature][cntLine];
  
              //phase2 process a file
               RandomAccessFile fileA = new RandomAccessFile("dataset.txt", "r");
               String[] line ;
               for(int i=0;i<cntLine;i++){
                    
                     line = fileA.readLine().split("\\s");
                    
                    for(int j=0;j<line.length;j++) {
          
                              feature[j][i] = Double.parseDouble(line[j]);
                              
                    }
               }
               
               //print it out
               Double[] theFeature = new Double[cntLine];
               for(int i=0;i<cntFeature;i++){
                   // System.out.println(""+(i+1)+" feature");
                    
                    //get the pre data into array(theFeature) for calculate SD and MEAN
                    for(int j=0;j<cntLine;j++) {
                        theFeature[j] = feature[i][j] ;
                    }
                    
                    //print out a 12 data in one feature
                    /**
                    for(int j=0;j<cntLine;j++) {
                        System.out.println(theFeature[j]);
                    }
               **/    
                    //print out a SD and Mean of 12 Data
                    double mean = calculateMean(theFeature) ;
                    double SD = calculateSD(theFeature) ;
                    //System.out.print("Mean : "); System.out.println(mean);
                    //System.out.print("SD :");System.out.println(SD);
                    //System.out.print("********************************************");
                    
                    
                    //improve a data by nomalization data
                    for(int j=0;j<cntLine;j++) {
                        feature[i][j] = (feature[i][j]-mean)/SD ;
                    }
                    
               } 
               
               //create new file for store an improve data
               File newFile = new File("newData.txt");
               PrintWriter pw = new PrintWriter(newFile);
               String row  = null;
               
               for(int i=0 ; i <cntLine ;i++) {
                   row = null;
                   for(int j=0;j<cntFeature;j++) {
                       if(j == (cntFeature-1)) {
                           
                           //case : is the last data of row
                           pw.println(row+feature[j][i]+"\n");
                       
                       } else {
                           if(row == null) {
                           row = ""+feature[j][i]+" ";
                           }else{ 
                           row += ""+feature[j][i]+" " ;
                           }
                       }
                   }
                   
               }
               pw.close();
               
         } catch(IOException e) {
             e.printStackTrace();
         }
    }
    
    public static double calculateSD(Double[] numArray)
    {
        double sum = 0.0, standardDeviation = 0.0;
        int length = numArray.length;

        for(double num : numArray) {
            sum += num;
        }

        double mean = sum/length;

        for(double num: numArray) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        return Math.sqrt(standardDeviation/length);
    }
    
    public static double calculateMean(Double[] numArray)
    {
        double sum = 0.0, standardDeviation = 0.0;
        int length = numArray.length;

        for(double num : numArray) {
            sum += num;
        }

        double mean = sum/length;

        return mean;
    }
    
    public static Double[] improveData(Double[] b) {
    
        //phase1 virable extract every data that nesscery from file
        int cntFeature =0 ;
        int cntLine =0 ;
        String[] a ;
        Double[] result =null ;
        
        try {
        
             //phase1
             //count  a number of features and lines
             File file = new File("dataset.txt");
             Scanner sc = new Scanner(file);
             while (sc.hasNextLine()) {
                    a =  sc.nextLine().split("\\s");
                    cntFeature = a.length;
                    cntLine++;
             } 
              sc.close();
              
              //create an array that contain a SD
              Double[] arrayOfSD = new Double[cntFeature];
              
              //create an array that contain a Mean
              Double[] arrayOfMean = new Double[cntFeature];
              
              //create an array that contain  a normalizated data
              Double[] normalData = new Double[cntFeature];
             
              //create an array that contain every feature
              Double[][] feature = new Double[cntFeature][cntLine];
  
              //phase2 process a file
               RandomAccessFile fileA = new RandomAccessFile("dataset.txt", "r");
               String[] line ;
               for(int i=0;i<cntLine;i++){
                    
                     line = fileA.readLine().split("\\s");
                    
                    for(int j=0;j<line.length;j++) {
          
                              feature[j][i] = Double.parseDouble(line[j]);
                              
                    }
               }
               
               //print it out
               Double[] theFeature = new Double[cntLine];
               for(int i=0;i<cntFeature;i++){
                    //System.out.println(""+(i+1)+" feature");
                    
                    //get the pre data into array(theFeature) for calculate SD and MEAN
                    for(int j=0;j<cntLine;j++) {
                        theFeature[j] = feature[i][j] ;
                    }
                    
                    //print out a 12 data in one feature
                    /**
                    for(int j=0;j<cntLine;j++) {
                        System.out.println(theFeature[j]);
                    }
               **/     
                    //print out a SD and Mean of 12 Data
                    arrayOfMean[i] = calculateMean(theFeature) ;
                    arrayOfSD[i] = calculateSD(theFeature) ;
               }
            
               for(int i=0; i < cntFeature ;i++) {
                   normalData[i] = (b[i] - arrayOfMean[i])/arrayOfSD[i];
               }
               
               result = normalData;
               
        } catch(IOException e) {
            e.printStackTrace();
        } 
        
        if(result == null) {
            Double[] arr = new Double[] {404.0,404.0};
            //System.out.println("Unreach");
            return arr ;
        } else {
            //System.out.println("reach!!");
            return result;
        }
        
    }
    

}
