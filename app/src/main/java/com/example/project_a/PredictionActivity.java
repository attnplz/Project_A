/*
    1. Get one image from external storage send to Image Processing class
    2. Prepare Feature Extraction for all images in external storage
 */

package com.example.project_a;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import com.example.project_a.image_classification.*;

public class PredictionActivity extends AppCompatActivity {

    private static final String TAG = "JUG";
    private int REQUEST_CODE_GALLERRY = 100;

    private TextView tv;
    Button btn_gallery, btn_predict_histogram, btn_predict_keypointsdescriptor;
    Uri imageUri;
    ImageView imv_original, imv_predicted_img;
    Mat sampledImgMat;
    Bitmap imageBitmap, grayBitmap;
    String extractedfeatures_hammer, extractedfeatures_scissors, extractedfeatures_paper;
    TextView tv_predicted_class;

    ArrayList<String> listOfAllImages;
    ArrayList<String> listOf_ProcessedImagesFile;
    Properties listOfAllImages_Class;
    MatOfKeyPoint keypointsBRISK, keypointsORB;
    Mat descriptorsBRISK, descriptorsORB;
    ArrayList<MatOfKeyPoint> keypointsORB_all_images;
    ArrayList<Mat> descriptorsORB_all_images;
    ArrayList<MatOfKeyPoint> keypointsBRISK_all_images;
    ArrayList<Mat> descriptorsBRISK_all_images;

    float[] color_histogram_feature, gray_histogram_feature, image_features;

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_prediction);

        btn_gallery = findViewById(R.id.btn_gallery);
        btn_predict_histogram = findViewById(R.id.btn_predict_histogram);
        btn_predict_keypointsdescriptor = findViewById(R.id.btn_predict_keypointsdescriptor);
        imv_original = findViewById(R.id.imv_original);
        imv_predicted_img = findViewById(R.id.imv_predicted_img);
        tv_predicted_class = findViewById(R.id.tv_predicted_class);

        if(OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"OpenCV loaded successfully", Toast.LENGTH_SHORT).show();
        }else{
            Toast.makeText(getApplicationContext(),"Could not load openCV", Toast.LENGTH_SHORT).show();
        }

        btn_gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, REQUEST_CODE_GALLERRY);
            }
        });

        btn_predict_histogram.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String PredictionMethod = "HISTOGRAM";        //HISTOGRAM, KEYPOINTSDESCRIPTOR
                ImageClassification imageClassification = new ImageClassification(image_features, extractedfeatures_hammer, extractedfeatures_scissors, extractedfeatures_paper,
                        keypointsORB, descriptorsORB, listOf_ProcessedImagesFile, listOfAllImages_Class, keypointsORB_all_images,descriptorsORB_all_images,PredictionMethod);
                String Determine_Class_histogram = imageClassification.Determine_Class_histogram;
                tv_predicted_class.setText("PREDICTED CLASS : " + Determine_Class_histogram);
                if (Determine_Class_histogram.equals("HAMMER")){
                    imv_predicted_img.setImageResource(R.drawable.rock);
                }
                if (Determine_Class_histogram.equals("PAPER")){
                    imv_predicted_img.setImageResource(R.drawable.paper);
                }
                if (Determine_Class_histogram.equals("SCISSORS")){
                    imv_predicted_img.setImageResource(R.drawable.scissors);
                }
            }
        });

        btn_predict_keypointsdescriptor.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String PredictionMethod = "KEYPOINTSDESCRIPTOR";        //HISTOGRAM, KEYPOINTSDESCRIPTOR
                ImageClassification imageClassification = new ImageClassification(image_features, extractedfeatures_hammer, extractedfeatures_scissors, extractedfeatures_paper,
                        keypointsORB, descriptorsORB, listOf_ProcessedImagesFile, listOfAllImages_Class, keypointsORB_all_images,descriptorsORB_all_images,PredictionMethod);
                String Determine_Class_histogram = imageClassification.Determine_Class_keypointsdescriptor;
                tv_predicted_class.setText("PREDICTED CLASS : " + Determine_Class_histogram);
                if (Determine_Class_histogram.equals("HAMMER")){
                    imv_predicted_img.setImageResource(R.drawable.rock);
                }
                if (Determine_Class_histogram.equals("PAPER")){
                    imv_predicted_img.setImageResource(R.drawable.paper);
                }
                if (Determine_Class_histogram.equals("SCISSORS")){
                    imv_predicted_img.setImageResource(R.drawable.scissors);
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        //Open image file from gallery
        if(requestCode == REQUEST_CODE_GALLERRY && resultCode == RESULT_OK && data != null) {
            //Setting imageUri from intent
            imageUri = data.getData();

            //Get path of imageUri
            String path = getPath(imageUri);

            //Get list of all images in specific folder
            String Current_image_path = path;
            listOfAllImages = getListOfAllImages(Current_image_path);       //All images except the current image

            //Set Property for all images
            listOfAllImages_Class = set_all_image_files_class();

            //Load image (Mat) from Uri
            sampledImgMat = loadImage(path);
            //sampledImgMat = loadImage(listOfAllImages.get(0));
            imageBitmap = convertMatToImageRGB(sampledImgMat);

            //Display image in imageView
            imv_original.setImageBitmap(imageBitmap);

            //*** EXTRACT FEATURE ****
            grayBitmap = convertToGray(imageBitmap);
            color_histogram_feature = get_histogram_rgb(sampledImgMat);
            gray_histogram_feature = get_histogram_gray(grayBitmap);

            //Get histogram features
            image_features = new float[color_histogram_feature.length + gray_histogram_feature.length];
            System.arraycopy(color_histogram_feature, 0, image_features, 0, color_histogram_feature.length);
            System.arraycopy(gray_histogram_feature,0,image_features,color_histogram_feature.length, gray_histogram_feature.length);

            //Get keypoints, descriptors (ORB)
            get_ORB_keypoint_descriptor(sampledImgMat);
            get_ORB_keypoint_descriptor_all_images(listOfAllImages);

            extractedfeatures_hammer = get_normalized_data_from_file("HAMMER");    //Hammer
            extractedfeatures_scissors = get_normalized_data_from_file("SCISSORS");    //Scissors
            extractedfeatures_paper = get_normalized_data_from_file("PAPER");    //Paper
        }
    }

    private ArrayList<String> getListOfAllImages(String Current_image_path){
        //https://stackoverflow.com/questions/18590514/loading-all-the-images-from-gallery-into-the-application-in-android
        Uri uri;
        Cursor cursor;
        int column_index_data, column_index_folder_name;
        ArrayList<String> temp_listOfAllImages = new ArrayList<String>();
        ArrayList<String> listOfAllImages = new ArrayList<String>();
        String absolutePathOfImage = null;
        uri = android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
        String[] projection = {MediaStore.MediaColumns.DATA,
                MediaStore.Images.Media.BUCKET_DISPLAY_NAME };

        cursor = PredictionActivity.this.getContentResolver().query(uri, projection, null,
                null, null);
        column_index_data = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA);
        column_index_folder_name = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME);

        while (cursor.moveToNext()) {
            absolutePathOfImage = cursor.getString(column_index_data);
            temp_listOfAllImages.add(absolutePathOfImage);
        }
        //Remove the current file
        for (int i=0; i<temp_listOfAllImages.size();i++){
            if (temp_listOfAllImages.get(i).equals(Current_image_path)==false){
                listOfAllImages.add(temp_listOfAllImages.get(i));
            }
        }
        return listOfAllImages;
    }

    private String getPath(Uri uri){
        if(uri == null){
            return null;
        }else{
            String[] projection = {MediaStore.Images.Media.DATA};
            Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
            if (cursor != null){
                int col_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
                cursor.moveToFirst();
                return cursor.getString(col_index);
            }
        }
        return uri.getPath();
    }

    private Mat loadImage(String path){
        Mat originImage = Imgcodecs.imread(path); //Image will be in BGR format
        Mat rgbImg = new Mat();

        //Convert BGR to RGB
        Imgproc.cvtColor(originImage,rgbImg, Imgproc.COLOR_BGR2RGB);

        Display display = getWindowManager().getDefaultDisplay();

        Point size = new Point();
        display.getSize(size);

        int mobile_width = size.x;
        int mobile_height = size.y;

        sampledImgMat = new Mat();
        double downSampleRatio = calculateSubSimpleSize(rgbImg, mobile_width, mobile_height);

        Imgproc.resize(rgbImg,sampledImgMat, new Size(), downSampleRatio, downSampleRatio, Imgproc.INTER_AREA);

        try {
            ExifInterface exif = new ExifInterface(path);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1);
            switch (orientation){
                case ExifInterface.ORIENTATION_ROTATE_90:
                    sampledImgMat = sampledImgMat.t();
                    Core.flip(sampledImgMat, sampledImgMat,1);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    sampledImgMat = sampledImgMat.t();
                    Core.flip(sampledImgMat, sampledImgMat,0);
                    break;
            }
        }catch(IOException e) {
            e.printStackTrace();
        }
        return sampledImgMat;
    }

    private double calculateSubSimpleSize(Mat src, int mobile_width, int mobile_height) {
        final int width = src.width();
        final int height = src.height();
        double inSampleSize = 1;

        if (height > mobile_height || width > mobile_width){
            //Calculate the ratio
            final double heightRatio = (double)mobile_height / (double)height;
            final double widthRatio = (double)mobile_width / (double)width;

            inSampleSize = heightRatio < widthRatio ? height : width;
        }
        return inSampleSize;
    }

    private Bitmap convertMatToImageRGB(Mat mat){
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(),mat.rows(),Bitmap.Config.RGB_565);
        //Convert mat to bitmap
        Utils.matToBitmap(mat, bitmap);
        return bitmap;
    }

    private Bitmap convertToGray(Bitmap imageBitmap){
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inDither = false;
        o.inSampleSize = 4;

        //Read size of Mat
        int width = imageBitmap.getWidth();
        int height = imageBitmap.getHeight();

        //Create Mat
        Mat Rgba = new Mat();       //Input
        Mat grayMat = new Mat();    //Output

        //Convert Bitmap into Mat
        Utils.bitmapToMat(imageBitmap, Rgba);

        //Image Processing
        Imgproc.cvtColor(Rgba, grayMat, Imgproc.COLOR_BGR2GRAY);

        //Convert Mat into bitmap value
        //Create grayBitmap
        Bitmap grayBitmap = Bitmap.createBitmap(width, height,Bitmap.Config.RGB_565);
        Utils.matToBitmap(grayMat, grayBitmap);

        return grayBitmap;
    }

    private float[] get_histogram_rgb(Mat image){
        //Matrix will hold the histogram values
        Mat hist = new Mat();

        //Number of Histogram bins
        int mHistSizeNum = 25;

        //A Matrix of one column and one row holding the number of histogram bins
        MatOfInt mHistSize = new MatOfInt(mHistSizeNum);

        //A float array to hold the histogram values
        float []mBuff = new float[mHistSizeNum];

        //histogram value holds 3-channel histogram value
        float []Hist = new float[mHistSizeNum * 3];

        //A matrix of one column and two rows holding the histogram range
        MatOfFloat histogramRanges = new MatOfFloat(0f, 256f);

        //A mask just in case you wanted to calculate the histogram for a specific area in the image
        Mat mask=new Mat();

        MatOfInt mChannels[] = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };

        // RGB
        float[] arr_hist = new float[0];

        for(int c=0; c<3; c++) {
            Imgproc.calcHist(Arrays.asList(image), mChannels[c], mask, hist, mHistSize, histogramRanges);

            //set a limit to the maximum histogram value, so you can display it on your device screen
            //Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);


            //get the histogram values for channel C, (hist --> mBuff)
            hist.get(0, 0, mBuff);

            //Normalization with size of image
            float sum_value = 0.0f;
            for(int i=0;i<mBuff.length;i++){
                mBuff[i] /= (image.rows()*image.cols());
                sum_value+=mBuff[i];
            }
            //Log.d(TAG, String.valueOf(sum_value));

            //Concatenate histogram values
            int fal = arr_hist.length;
            int sal = mBuff.length;
            float[] result = new float[fal + sal];
            System.arraycopy(arr_hist, 0,result,0,fal);
            System.arraycopy(mBuff,0,result,fal,sal);
            arr_hist = result;
        }
        //Return histogram values
        return arr_hist;
    }

    private float[] get_histogram_gray(Bitmap grayBitmap){
        Mat sourceMat = new Mat();
        Utils.bitmapToMat(grayBitmap, sourceMat);

        Size sourceSize = sourceMat.size();

        int histogramSize = 25;
        MatOfInt hisSize = new MatOfInt(histogramSize);

        List<Mat> channels = new ArrayList<>();

        MatOfFloat range = new MatOfFloat(0f, 256f);
        MatOfFloat histRange = new MatOfFloat(range);

        Core.split(sourceMat, channels);

        MatOfInt[] allChannel = new MatOfInt[]{new MatOfInt(0)};

        Mat matB = new Mat(sourceSize, sourceMat.type());

        Imgproc.calcHist(channels, allChannel[0], new Mat(), matB, hisSize, histRange);

        //A float array to hold the histogram values
        float []mBuff = new float[histogramSize];

        matB.get(0,0,mBuff);

        //Normalize channel
        //Core.normalize(matB, matB, graphMat.height(), 0, Core.NORM_INF);

        //Normalization with size of image
        float sum_value = 0.0f;
        for(int i=0;i<mBuff.length;i++){
            mBuff[i] /= (sourceMat.rows()*sourceMat.cols());
            sum_value+=mBuff[i];
        }
        //Log.d(TAG, String.valueOf(sum_value));

        float[] arr_hist = mBuff;

        //Return histogram values
        return arr_hist;
    }

    private void get_ORB_keypoint_descriptor(Mat sampledImgMat){
        keypointsORB = new MatOfKeyPoint();
        descriptorsORB = new Mat();

        FeatureDetector detector = null;
        DescriptorExtractor descriptorExtractor = null;
        detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        detector.detect(sampledImgMat, keypointsORB);
        descriptorExtractor.compute(sampledImgMat, keypointsORB, descriptorsORB);
    }

    private void get_BRISK_keypoint_descriptor(Mat sampledImgMat){
        keypointsBRISK = new MatOfKeyPoint();
        descriptorsBRISK = new Mat();

        FeatureDetector detector = null;
        DescriptorExtractor descriptorExtractor = null;
        detector = FeatureDetector.create(FeatureDetector.BRISK);
        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
        detector.detect(sampledImgMat, keypointsBRISK);
        descriptorExtractor.compute(sampledImgMat, keypointsBRISK, descriptorsBRISK);
    }

    private String get_normalized_data_from_file(String className){
        String data = "";
        StringBuffer sBuffer = new StringBuffer();
        InputStream is;

        if (className.equals("HAMMER")){
            is = this.getResources().openRawResource(R.raw.hammer);
        } else if (className.equals("SCISSORS")){
            is = this.getResources().openRawResource(R.raw.scissors);
        }else if (className.equals("PAPER")){
            is = this.getResources().openRawResource(R.raw.paper);
        }else {
            is = this.getResources().openRawResource(R.raw.extractedfeatures);
        }

        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        if(is != null){
            try{
                while((data = reader.readLine())!=null){
                    sBuffer.append(data + "n");
                }
                is.close();
            }catch(Exception e){
                e.printStackTrace();
            }
        }
        return sBuffer.toString();
    }

    private Properties set_all_image_files_class(){
        Properties prop = new Properties();
        prop.setProperty("20200422_042737.jpg","SCISSORS");
        prop.setProperty("20200422_043006.jpg","HAMMER");
        return prop;
    }

    private void get_ORB_keypoint_descriptor_all_images(ArrayList<String> listOfAllImages){
        keypointsORB_all_images = new ArrayList<>();
        descriptorsORB_all_images = new ArrayList<>();
        listOf_ProcessedImagesFile = new ArrayList<>();

        for (int i=0; i<listOfAllImages.size();i++){
            String path = listOfAllImages.get(i);
            //Load image path
            Mat sampledImgMat = loadImage(path);

            //Process to get keypoint and desriptor for each image
            MatOfKeyPoint keypointsORB = new MatOfKeyPoint();
            Mat descriptorsORB = new Mat();
            FeatureDetector detector = null;
            DescriptorExtractor descriptorExtractor = null;
            detector = FeatureDetector.create(FeatureDetector.ORB);
            descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
            detector.detect(sampledImgMat, keypointsORB);
            descriptorExtractor.compute(sampledImgMat, keypointsORB, descriptorsORB);
            //Return value
            listOf_ProcessedImagesFile.add(path);
            keypointsORB_all_images.add(keypointsORB);
            descriptorsORB_all_images.add(descriptorsORB);
        }
    }

    private void get_BRISK_keypoint_descriptor_all_images(ArrayList<String> listOfAllImages){
        keypointsBRISK_all_images = new ArrayList<>();
        descriptorsBRISK_all_images = new ArrayList<>();

        for (int i=0; i<listOfAllImages.size();i++){
            String path = listOfAllImages.get(i);
            //Load image path
            Mat sampledImgMat = loadImage(path);

            //Process to get keypoint and desriptor for each image
            MatOfKeyPoint keypointsBRISK = new MatOfKeyPoint();
            Mat descriptorsBRISK = new Mat();
            FeatureDetector detector = null;
            DescriptorExtractor descriptorExtractor = null;
            detector = FeatureDetector.create(FeatureDetector.BRISK);
            descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
            detector.detect(sampledImgMat, keypointsBRISK);
            descriptorExtractor.compute(sampledImgMat, keypointsBRISK, descriptorsBRISK);
            //Return value
            keypointsBRISK_all_images.add(keypointsBRISK);
            descriptorsBRISK_all_images.add(descriptorsBRISK);
        }
    }
}

