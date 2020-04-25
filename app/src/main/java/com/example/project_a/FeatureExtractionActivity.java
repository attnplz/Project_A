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
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class FeatureExtractionActivity extends AppCompatActivity {

    private static final String TAG = "JUG";
    ImageView imv_original, imv_imgproc;
    Button btn_gallery, btn_histogram, btn_gray_histogram, btn_orb, btn_brisk, btn_convex;
    Bitmap imageBitmap, grayBitmap, contourBitmap;
    Mat sampledImgMat;
    Uri imageUri;
    TextView tv_feature_value;

    private int keypointsObject;
    private int REQUEST_CODE_GALLERRY = 100;
    private boolean src1Selected = false;

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
        setContentView(R.layout.activity_feature_extraction);

        btn_gallery = findViewById(R.id.btn_gallery);
        btn_gray_histogram = findViewById(R.id.btn_gray_histogram);
        btn_histogram = findViewById(R.id.btn_histogram);
        btn_orb = findViewById(R.id.btn_orb);
        btn_brisk = findViewById(R.id.btn_brisk);
        btn_convex = findViewById(R.id.btn_convex);

        imv_original = findViewById(R.id.imv_original);
        imv_imgproc = findViewById(R.id.imv_imgproc);

        tv_feature_value = findViewById(R.id.tv_feature_value);

        keypointsObject = -1;

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

        btn_histogram.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                histogram_grb2(sampledImgMat);
                imv_imgproc.setImageBitmap(imageBitmap);
            }
        });

        btn_gray_histogram.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                convertToGray(v);
                histogram_gray(v);
            }
        });

        btn_orb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                executeORB();
            }
        });

        btn_brisk.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                executeBRISK();
            }
        });

        btn_convex.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                List<MatOfPoint> contoursList = getImageContoursList();
                getConvexFeature(contoursList);
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
            imv_original.setImageBitmap(imageBitmap);
            src1Selected = true;
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

    private void convertToGray(View v){
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
        grayBitmap = Bitmap.createBitmap(width, height,Bitmap.Config.RGB_565);
        Utils.matToBitmap(grayMat, grayBitmap);

        imv_imgproc.setImageBitmap(grayBitmap);
    }

    private void histogram_grb(Mat sampledImgMat){
        Mat sourceMat = new Mat();
        Utils.bitmapToMat(imageBitmap, sourceMat);

        Size sourceSize = sourceMat.size();


        int histogramSize = 256;
        MatOfInt hisSize = new MatOfInt(histogramSize);

        Mat destinationMat = new Mat();
        List<Mat> channels = new ArrayList<>();

        MatOfFloat range = new MatOfFloat(0f, 255f);
        MatOfFloat histRange = new MatOfFloat(range);

        Core.split(sourceMat, channels);

        MatOfInt[] allChannel = new MatOfInt[]{new MatOfInt(0), new MatOfInt(1), new MatOfInt(2)};
        Scalar[] colorScalar = new Scalar[]{new Scalar(220, 0, 0, 255), new Scalar(0, 220, 0, 255), new Scalar(0, 0, 220, 255)};

        Mat matB = new Mat(sourceSize, sourceMat.type());
        Mat matG = new Mat(sourceSize, sourceMat.type());
        Mat matR = new Mat(sourceSize, sourceMat.type());

        Imgproc.calcHist(channels, allChannel[0], new Mat(), matB, hisSize, histRange);
        Imgproc.calcHist(channels, allChannel[1], new Mat(), matG, hisSize, histRange);
        Imgproc.calcHist(channels, allChannel[2], new Mat(), matR, hisSize, histRange);

        //int graphHeight = 300;
        //int graphWidth = 200;

        int graphHeight = 100;
        int graphWidth = 50;

        int binWidth = 3;

        Mat graphMat = new Mat(graphHeight, graphWidth, CvType.CV_8UC3, new Scalar(0, 0, 0));

        //Normalize channel
        Core.normalize(matB, matB, graphMat.height(), 0, Core.NORM_INF);
        Core.normalize(matG, matG, graphMat.height(), 0, Core.NORM_INF);
        Core.normalize(matR, matR, graphMat.height(), 0, Core.NORM_INF);

        //convert pixel value to point and draw line with points
        for(int i = 0; i < histogramSize; i++){
            Point bPoint1 = new Point(binWidth * (i - 1), (int) (graphHeight - Math.round(matB.get(i - 1, 0)[0])));
            Point bPoint2 = new Point(binWidth * i, (int) (graphHeight - Math.round(matB.get(i, 0)[0])));
            Core.line(graphMat, bPoint1, bPoint2, new Scalar(220, 0, 0, 255), 3, 8, 0);

            Point gPoint1 = new Point(binWidth * (i - 1), (int) (graphHeight - Math.round(matG.get(i - 1, 0)[0])));
            Point gPoint2 = new Point(binWidth * i, (int) (graphHeight - Math.round(matG.get(i, 0)[0])));
            Core.line(graphMat, gPoint1, gPoint2, new Scalar(0, 220, 0, 255), 3, 8, 0);

            Point rPoint1 = new Point(binWidth * (i - 1), (int) (graphHeight - Math.round(matR.get(i - 1, 0)[0])));
            Point rPoint2 = new Point(binWidth * i, (int) (graphHeight - Math.round(matR.get(i, 0)[0])));
            Core.line(graphMat, rPoint1, rPoint2, new Scalar(0, 0, 220, 255), 3, 8, 0);
        }

        //convert Mat to bitmap
        Bitmap graphBitmap = Bitmap.createBitmap(graphMat.cols(), graphMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(graphMat, graphBitmap);

        // show histogram
        imv_imgproc.setImageBitmap(graphBitmap);
    }

    private void histogram_grb2(Mat image) {
        //Matrix will hold the histogram values
        Mat hist = new Mat();

        //Number of Histogram bins
        int mHistSizeNum = 25;

        //A matrix of one column and one row holding the number of histogram bins
        MatOfInt mHistSize = new MatOfInt(mHistSizeNum);

        //A float array to hold the histogram values
        float []mBuff = new float[mHistSizeNum];

        //A matrix of one column and two rows holding the histogram range
        MatOfFloat histogramRanges = new MatOfFloat(0f, 256f);

        //A mask just in case you wanted to calculate the histogram for a specific area in the image
        Mat mask=new Mat();

        MatOfInt mChannels[] = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };

        // RGB
        StringBuilder builder = new StringBuilder();
        float[] arr_f;
        builder.append("Feature value : ");
        for(int c=0; c<3; c++) {
            builder.append("channel : ").append(c+1).append(" --- ");
            Imgproc.calcHist(Arrays.asList(image), mChannels[c], mask, hist, mHistSize, histogramRanges);

            //set a limit to the maximum histogram value, so you can display it on your device screen
            //Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);

            //get the histogram values for channel C
            hist.get(0, 0, mBuff);

            arr_f = mBuff;
            for(float s : arr_f){
                builder.append(s).append(" ");
            }
        }
        tv_feature_value.setText(builder.toString());
    }

    private void histogram_gray(View v){
        Mat sourceMat = new Mat();
        Utils.bitmapToMat(grayBitmap, sourceMat);

        Size sourceSize = sourceMat.size();

        int histogramSize = 25;
        MatOfInt hisSize = new MatOfInt(histogramSize);

        List<Mat> channels = new ArrayList<>();

        MatOfFloat range = new MatOfFloat(0f, 255f);
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

        StringBuilder builder = new StringBuilder();
        float[] arr_f = mBuff;
        builder.append("Feature value : ");
        builder.append("channel : ").append(1).append(" --- ");
        for(float s : arr_f){
            builder.append(s).append(" ");
        }
        tv_feature_value.setText(builder.toString());
    }

    private void executeORB(){

        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();

        FeatureDetector detector;
        DescriptorExtractor descriptorExtractor;

        detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        detector.detect(sampledImgMat, keypoints);

        keypointsObject = keypoints.toArray().length;

        descriptorExtractor.compute(sampledImgMat,keypoints,descriptors);

        tv_feature_value.setText("Number of keypoints : " + keypointsObject);
    }

    private void executeBRISK(){
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();

        FeatureDetector detector;
        DescriptorExtractor descriptorExtractor;

        detector = FeatureDetector.create(FeatureDetector.BRISK);
        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);

        detector.detect(sampledImgMat, keypoints);

        keypointsObject = keypoints.toArray().length;

        descriptorExtractor.compute(sampledImgMat,keypoints,descriptors);
        tv_feature_value.setText("Number of keypoints : " + keypointsObject);
    }

    private List<MatOfPoint> getImageContoursList(){
        //Read image to Bitmap, then convert to Mat
        //Read size of Mat
        int width = imageBitmap.getWidth();
        int height = imageBitmap.getHeight();

        //Create Mat
        Mat Rgba = new Mat();

        //Convert Bitmap into Mat
        Utils.bitmapToMat(imageBitmap, Rgba);

        Mat cannyEdges = new Mat();
        Mat hierarchy = new Mat();
        Mat contours = new Mat();

        //Create A list to store all the contours
        List<MatOfPoint> contourList = new ArrayList<MatOfPoint>();

        //Image Processing
        Imgproc.cvtColor(Rgba, Rgba, Imgproc.COLOR_BGR2GRAY);
        //Thresholding
        Imgproc.threshold(Rgba, Rgba, 50,255.0,Imgproc.THRESH_BINARY);
        //Morphology
        Mat kernalErode = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7,7));
        Mat kernalDilate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));
        Imgproc.erode(Rgba, Rgba, kernalErode);
        Imgproc.dilate(Rgba, Rgba, kernalDilate);
        Imgproc.dilate(Rgba, Rgba, kernalDilate);

        //Process Canny edge and store in cannyEdge
        Imgproc.Canny(Rgba, cannyEdges, 10, 100);

        //Finding contours
        Imgproc.findContours(cannyEdges, contourList, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        //Create a Mat for store contour
        contours.create(cannyEdges.rows(), cannyEdges.cols(), CvType.CV_8UC3);

        //Draw contour
        Imgproc.drawContours(contours, contourList, -1, new Scalar(255,255,255), -1);

        //Converting Mat back to Bitmap
        //contourBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(contours, contourBitmap);

        //imageView.setImageBitmap(contourBitmap);

        return contourList;
    }

    private void getConvexFeature(List<MatOfPoint> contours){
        //get image

        // Find the convex hull
        List<MatOfInt> hull = new ArrayList<MatOfInt>();
        for(int i=0; i < contours.size(); i++){
            hull.add(new MatOfInt());
        }
        for(int i=0; i < contours.size(); i++){
            Imgproc.convexHull(contours.get(i), hull.get(i));
        }

        // Convert MatOfInt to MatOfPoint for drawing convex hull

        // Loop over all contours
        List<Point[]> hullpoints = new ArrayList<Point[]>();
        for(int i=0; i < hull.size(); i++){
            Point[] points = new Point[hull.get(i).rows()];

            // Loop over all points that need to be hulled in current contour
            for(int j=0; j < hull.get(i).rows(); j++){
                int index = (int)hull.get(i).get(j, 0)[0];

                points[j] = new Point((int)contours.get(i).get(index, 0)[0], (int)contours.get(i).get(index, 0)[1]);
            }

            hullpoints.add(points);
        }

        // Convert Point arrays into MatOfPoint
        List<MatOfPoint> hullmop = new ArrayList<MatOfPoint>();
        for(int i=0; i < hullpoints.size(); i++){
            MatOfPoint mop = new MatOfPoint();

            //mop.fromArray(hullpoints.get(i));

            for (int j = 0; j<hullpoints.get(i).length; j++) {
                ArrayList<org.opencv.core.Point> pointsOrdered = new ArrayList<org.opencv.core.Point>();
                pointsOrdered.add(new org.opencv.core.Point(hullpoints.get(i)[j].x, hullpoints.get(i)[j].y));

                mop.fromList(pointsOrdered);

                hullmop.add(mop);
            }
        }

        // Draw contours + hull results
        //Mat overlay = new Mat(binaryImage.size(), CvType.CV_8UC3);

        Mat binaryImage = new Mat();
        //Convert bitmap to Mat
        Utils.bitmapToMat(imageBitmap, binaryImage);

        Mat overlay = new Mat(binaryImage.size(), CvType.CV_8UC3);
        Scalar color = new Scalar(0, 255, 0);   // Green
        for(int i=0; i < contours.size(); i++){
            //Imgproc.drawContours(overlay, contours, i, color);
            Imgproc.drawContours(overlay, hullmop, i, color);
        }

        Bitmap temp_bitmap = convertMatToImageRGB(overlay);
        imv_imgproc.setImageBitmap(temp_bitmap);
    }

}
