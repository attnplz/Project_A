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
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.example.project_a.image_classification.DistanceCalculation;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BOWImgDescriptorExtractor;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class CompareImages extends AppCompatActivity {
    private static final String TAG = "JUG";

    Button btn_gallery1, btn_gallery2, btn_histogram_rgb, btn_histogram_gray, btn_brisk, btn_orb;
    ImageView imv_gallery1, imv_gallery2, imv_feature_image, imv_feature_image2;

    Mat sampledImgMat, sampledImgMat1, sampledImgMat2;
    Bitmap imageBitmap, imageBitmap1, imageBitmap2, grayBitmap1, grayBitmap2, mathchBitmapORB;
    Boolean src1selected, src2selected;

    TextView tv_title_feature, tv_feature_description;

    Uri imageUri;
    private int keypointsObject1, keypointsObject2, keypointMatches;
    private final int MAX_MATCHES = 50;

    private int REQUEST_CODE_GALLERRY = 100;

    float[] histogram_img1, histogram_img2, histogram_gray_img1, histogram_gray_img2;
    float[] distance_orb;
    float[] distance_brisk;
    float sum_distance = 0.0f;

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
        setContentView(R.layout.activity_compare_images);

        btn_gallery1 = findViewById(R.id.btn_gallery1);
        btn_gallery2 = findViewById(R.id.btn_gallery2);
        btn_histogram_rgb = findViewById(R.id.btn_histogram_rgb);
        btn_histogram_gray = findViewById(R.id.btn_histogram_gray);
        btn_brisk = findViewById(R.id.btn_brisk);
        btn_orb = findViewById(R.id.btn_orb);
        tv_title_feature = findViewById(R.id.tv_title_feature);
        tv_feature_description = findViewById(R.id.tv_feature_description);

        imv_gallery1 = findViewById(R.id.imv_gallery1);
        imv_gallery2 = findViewById(R.id.imv_gallery2);
        imv_feature_image = findViewById(R.id.imv_feature_image);
        imv_feature_image2 = findViewById(R.id.imv_feature_image2);

        src1selected = false;
        src2selected = false;

        if(OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"OpenCV loaded successfully", Toast.LENGTH_SHORT).show();
        }else{
            Toast.makeText(getApplicationContext(),"Could not load openCV", Toast.LENGTH_SHORT).show();
        }

        btn_gallery1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                src1selected = true;
                startActivityForResult(intent, REQUEST_CODE_GALLERRY);
            }
        });
        btn_gallery2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                src2selected = true;
                startActivityForResult(intent, REQUEST_CODE_GALLERRY);
            }
        });
        btn_histogram_rgb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imv_feature_image2.setVisibility(View.VISIBLE);
                imv_gallery1.setImageBitmap(imageBitmap1);
                imv_gallery2.setImageBitmap(imageBitmap2);

                display_histogram(imageBitmap1, 3, imv_feature_image);
                display_histogram(imageBitmap2, 3, imv_feature_image2);

                tv_title_feature.setText("Histogram of RGB image");
                histogram_img1 = get_histogram_rgb(sampledImgMat1);
                histogram_img2 = get_histogram_rgb(sampledImgMat2);
                tv_feature_description.setText(displayDistanceDetails(histogram_img1, histogram_img2));
                Toast.makeText(getApplicationContext(),"Extracting histogram feature successfully", Toast.LENGTH_SHORT).show();
            }
        });
        btn_histogram_gray.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imv_feature_image2.setVisibility(View.VISIBLE);
                grayBitmap1 = convertToGray(imageBitmap1);
                grayBitmap2 = convertToGray(imageBitmap2);
                imv_gallery1.setImageBitmap(grayBitmap1);
                imv_gallery2.setImageBitmap(grayBitmap2);

                display_histogram(grayBitmap1, 1, imv_feature_image);
                display_histogram(grayBitmap2, 1, imv_feature_image2);

                tv_title_feature.setText("Histogram of grayscale image");
                histogram_gray_img1 = get_histogram_gray(grayBitmap1);
                histogram_gray_img2 = get_histogram_gray(grayBitmap2);
                tv_feature_description.setText(displayDistanceDetails(histogram_gray_img1, histogram_gray_img2));
                Toast.makeText(getApplicationContext(),"Extracting histogram feature successfully", Toast.LENGTH_SHORT).show();
            }
        });
        btn_brisk.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imv_feature_image2.setVisibility(View.INVISIBLE);
                imv_gallery1.setImageBitmap(imageBitmap1);
                imv_gallery2.setImageBitmap(imageBitmap2);

                tv_title_feature.setText("Keypoints Matching BRISK feature");
                mathchBitmapORB = matching_images(sampledImgMat1, sampledImgMat2, "BRISK");
                imv_feature_image.setImageBitmap(mathchBitmapORB);
                StringBuilder builder = new StringBuilder();
                builder.append("Matching BRISK feature detail :");
                builder.append(System.getProperty("line.separator"));
                builder.append("The number of keypoints on the first image : ");
                builder.append(keypointsObject1);
                builder.append(System.getProperty("line.separator"));
                builder.append("The number of keypoints on the second image : ");
                builder.append(keypointsObject2);
                builder.append(System.getProperty("line.separator"));
                builder.append("Total distance value : ");
                builder.append(sum_distance);
                tv_feature_description.setText(builder.toString());
                Toast.makeText(getApplicationContext(),"Matching BRISK feature successfully", Toast.LENGTH_SHORT).show();
            }
        });
        btn_orb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imv_feature_image2.setVisibility(View.INVISIBLE);
                imv_gallery1.setImageBitmap(imageBitmap1);
                imv_gallery2.setImageBitmap(imageBitmap2);

                tv_title_feature.setText("Keypoints Matching ORB feature");
                mathchBitmapORB = matching_images(sampledImgMat1, sampledImgMat2, "ORB");
                imv_feature_image.setImageBitmap(mathchBitmapORB);
                StringBuilder builder = new StringBuilder();
                builder.append("Matching ORB feature detail :");
                builder.append(System.getProperty("line.separator"));
                builder.append("The number of keypoints on the first image : ");
                builder.append(keypointsObject1);
                builder.append(System.getProperty("line.separator"));
                builder.append("The number of keypoints on the second image : ");
                builder.append(keypointsObject2);
                builder.append(System.getProperty("line.separator"));
                builder.append("Total distance value : ");
                builder.append(sum_distance);
                tv_feature_description.setText(builder.toString());
                Toast.makeText(getApplicationContext(),"Matching ORB feature successfully", Toast.LENGTH_SHORT).show();
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

            //Load image (Mat) from Uri
            sampledImgMat = loadImage(path);
            imageBitmap = convertMatToImageRGB(sampledImgMat);

            //Display image in imageView
            if (src1selected) {
                imageBitmap1 = imageBitmap;
                sampledImgMat1 = sampledImgMat;
                imv_gallery1.setImageBitmap(imageBitmap1);
                src1selected = false;
            }
            if (src2selected) {
                imageBitmap2 = imageBitmap;
                sampledImgMat2 = sampledImgMat;
                imv_gallery2.setImageBitmap(imageBitmap2);
                src2selected = false;
            }
        }
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

    private Bitmap matching_images(Mat sampledImgMat1, Mat sampledImgMat2, String Method){

        FeatureDetector detector = null;
        MatOfKeyPoint keypoints1, keypoints2;
        DescriptorExtractor descriptorExtractor = null;
        Mat descriptors1, descriptors2;
        DescriptorMatcher descriptorMatcher = null;
        MatOfDMatch matches = new MatOfDMatch();
        keypoints1 = new MatOfKeyPoint();
        keypoints2 = new MatOfKeyPoint();
        descriptors1 = new Mat();
        descriptors2 = new Mat();

        if(Method.equals("ORB")){
            detector = FeatureDetector.create(FeatureDetector.ORB);
            descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
            descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        }else if(Method.equals("BRISK")){
            detector = FeatureDetector.create(FeatureDetector.BRISK);
            descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
            descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        }

        //Get keypoints of 2 images
        detector.detect(sampledImgMat1, keypoints1);
        detector.detect(sampledImgMat2, keypoints2);

        keypointsObject1 = keypoints1.toArray().length;
        keypointsObject2 = keypoints2.toArray().length;

        //Get descriptorห
        descriptorExtractor.compute(sampledImgMat1, keypoints1, descriptors1);
        descriptorExtractor.compute(sampledImgMat2, keypoints2, descriptors2);

        //Matching 2 descriptorห.
        descriptorMatcher.match(descriptors1, descriptors2, matches);
        keypointMatches = matches.toArray().length;

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

        //Create Matching image
        Mat matchedImgMat = drawMatches(sampledImgMat1, keypoints1, sampledImgMat2, keypoints2, matches, false);

        //Return Bitmap from Mat
        Bitmap image1 = Bitmap.createBitmap(matchedImgMat.cols(), matchedImgMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matchedImgMat, image1);
        Imgproc.cvtColor(matchedImgMat, matchedImgMat, Imgproc.COLOR_BGR2RGB);

        return image1;
    }

    static Mat drawMatches(Mat img1, MatOfKeyPoint key1, Mat img2, MatOfKeyPoint key2, MatOfDMatch matches, boolean imageOnly) {
        Mat out = new Mat();
        Mat im1 = new Mat();
        Mat im2 = new Mat();
        Imgproc.cvtColor(img1, im1, Imgproc.COLOR_BGR2RGB);
        Imgproc.cvtColor(img2, im2, Imgproc.COLOR_BGR2RGB);
        if (imageOnly){
            MatOfDMatch emptyMatch = new MatOfDMatch();
            MatOfKeyPoint emptyKey1 = new MatOfKeyPoint();
            MatOfKeyPoint emptyKey2 = new MatOfKeyPoint();
            Features2d.drawMatches(im1, emptyKey1, im2, emptyKey2, emptyMatch, out);
        } else {
            Features2d.drawMatches(im1, key1, im2, key2, matches, out);
        }
        Bitmap bmp = Bitmap.createBitmap(out.cols(), out.rows(), Bitmap.Config.ARGB_8888);
        Imgproc.cvtColor(out, out, Imgproc.COLOR_BGR2RGB);
        return out;
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

        float[] arr_hist = mBuff;

        //Return histogram values
        return arr_hist;
    }

    private void display_histogram(Bitmap imageBitmap, int numChannel, ImageView imv_feature_image){
        int histogramSize = 256;
        int binWidth = 3;
        int graphHeight = imv_feature_image.getHeight();
        int graphWidth = imv_feature_image.getWidth();

        //convert from Bitmap to Mat
        Mat sourceMat = new Mat();
        Size sourceSize = sourceMat.size();
        Utils.bitmapToMat(imageBitmap, sourceMat);

        //Create Mat
        MatOfFloat range = new MatOfFloat(0f, 255f);
        MatOfFloat histRange = new MatOfFloat(range);
        MatOfInt hisSize = new MatOfInt(histogramSize);

        //Split Source Mat to 3 channels
        Mat destinationMat = new Mat();
        List<Mat> channels = new ArrayList<>();
        Core.split(sourceMat, channels);

        if (numChannel==3) {
            MatOfInt[] allChannel = new MatOfInt[]{new MatOfInt(0), new MatOfInt(1), new MatOfInt(2)};
            Scalar[] colorScalar = new Scalar[]{new Scalar(220, 0, 0, 255), new Scalar(0, 220, 0, 255), new Scalar(0, 0, 220, 255)};

            Mat matB = new Mat(sourceSize, sourceMat.type());
            Mat matG = new Mat(sourceSize, sourceMat.type());
            Mat matR = new Mat(sourceSize, sourceMat.type());
            Mat graphMat = new Mat(graphHeight, graphWidth, CvType.CV_8UC3, new Scalar(0, 0, 0));

            //Get histogram values for each channel
            Imgproc.calcHist(channels, allChannel[0], new Mat(), matB, hisSize, histRange);
            Imgproc.calcHist(channels, allChannel[1], new Mat(), matG, hisSize, histRange);
            Imgproc.calcHist(channels, allChannel[2], new Mat(), matR, hisSize, histRange);

            //Normalize channel
            Core.normalize(matB, matB, graphMat.height(), 0, Core.NORM_INF);
            Core.normalize(matG, matG, graphMat.height(), 0, Core.NORM_INF);
            Core.normalize(matR, matR, graphMat.height(), 0, Core.NORM_INF);

            //convert pixel value to point and draw line with points
            for (int i = 0; i < histogramSize; i++){
                //Blue component
                org.opencv.core.Point bPoint1 = new org.opencv.core.Point(binWidth * (i - 1), (graphHeight - (int)Math.round(matB.get(i - 1, 0)[0])));
                org.opencv.core.Point bPoint2 = new org.opencv.core.Point(binWidth * i, (graphHeight - (int)Math.round(matB.get(i, 0)[0])));
                Imgproc.line(graphMat, bPoint1, bPoint2, colorScalar[0],2);
                //Green component
                org.opencv.core.Point gPoint1 = new org.opencv.core.Point(binWidth * (i - 1), (graphHeight - (int)Math.round(matG.get(i - 1, 0)[0])));
                org.opencv.core.Point gPoint2 = new org.opencv.core.Point(binWidth * i, (graphHeight - (int)Math.round(matG.get(i, 0)[0])));
                Imgproc.line(graphMat, gPoint1, gPoint2, colorScalar[1], 2);
                //Red component
                org.opencv.core.Point rPoint1 = new org.opencv.core.Point(binWidth * (i - 1), (graphHeight - (int)Math.round(matR.get(i - 1, 0)[0])));
                org.opencv.core.Point rPoint2 = new org.opencv.core.Point(binWidth * i, (graphHeight - (int)Math.round(matR.get(i, 0)[0])));
                Imgproc.line(graphMat, rPoint1, rPoint2, colorScalar[2], 2);
            }

            //convert Mat to Bitmap
            Bitmap graphBitmap = Bitmap.createBitmap(graphMat.cols(), graphMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(graphMat, graphBitmap);
            imv_feature_image.setImageBitmap(graphBitmap);
        }else if(numChannel==1){
            MatOfInt[] allChannel = new MatOfInt[]{new MatOfInt(0)};
            Scalar[] grayScalar = new Scalar[]{new Scalar(255, 255, 255, 255)};

            Mat matGray = new Mat(sourceSize, sourceMat.type());
            Mat graphMat = new Mat(graphHeight, graphWidth, CvType.CV_8UC3, new Scalar(0, 0, 0));

            //Get histogram value for a channel
            Imgproc.calcHist(channels, allChannel[0], new Mat(), matGray, hisSize, histRange);

            //Normalize channel
            Core.normalize(matGray, matGray, graphMat.height(), 0, Core.NORM_INF);

            //convert pixel value to point and draw line with points
            for (int i = 0; i < histogramSize; i++) {
                //White component
                org.opencv.core.Point bPoint1 = new org.opencv.core.Point(binWidth * (i - 1), (graphHeight - (int)Math.round(matGray.get(i - 1, 0)[0])));
                org.opencv.core.Point bPoint2 = new org.opencv.core.Point(binWidth * i, (graphHeight - (int)Math.round(matGray.get(i, 0)[0])));
                Imgproc.line(graphMat, bPoint1, bPoint2, grayScalar[0],2);
            }

            //convert Mat to Bitmap
            Bitmap graphBitmap = Bitmap.createBitmap(graphMat.cols(), graphMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(graphMat, graphBitmap);
            imv_feature_image.setImageBitmap(graphBitmap);
        }
    }

    private String displayDistanceDetails(float[] hist1, float[] hist2){
        StringBuilder builder = new StringBuilder();

        DistanceCalculation distance = new DistanceCalculation(hist1, hist2);
        float euclidean = distance.euclideanDistance(hist1, hist2);
        float intersection = distance.intersectionDistance(hist1, hist2);
        float correlation = distance.correlationDistance(hist1, hist2);
        float chisquare = distance.chiSquareDistance(hist1, hist2);

        builder.append("Distance values : ");
        builder.append(System.getProperty("line.separator"));
        builder.append("Euclidean Distance values : ");
        builder.append(euclidean);
        builder.append(System.getProperty("line.separator"));
        builder.append("Histogram intersection values : ");
        builder.append(intersection);
        builder.append(System.getProperty("line.separator"));
        builder.append("Histogram correlation values : ");
        builder.append(correlation);
        builder.append(System.getProperty("line.separator"));
        builder.append("Histogram Chi-square values : ");
        builder.append(chisquare);

        return builder.toString();
    }
}


