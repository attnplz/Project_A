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
import android.os.AsyncTask;
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
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.MSER;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.features2d.FeatureDetector.MSER;

public class FeatureExtractionActivity extends AppCompatActivity {

    private static final String TAG = "JUG";
    ImageView imv_original, imv_imgproc;
    Button btn_gallery, btn_histogram, btn_gray_histogram, getBtn_histogram;
    Bitmap imageBitmap, grayBitmap;
    Mat sampledImgMat;
    Uri imageUri;
    TextView tv_feature_value;

    private int REQUEST_CODE_GALLERRY = 100;
    private boolean src1Selected = false;
    private int keypointsObject1;

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

        imv_original = findViewById(R.id.imv_original);
        imv_imgproc = findViewById(R.id.imv_imgproc);

        tv_feature_value = findViewById(R.id.tv_feature_value);

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
                histogram_gray();
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

//        StringBuilder builder = new StringBuilder();
//        String[] arr = {"These","are","some","words"};
//        for (String s : arr) {
//            builder.append(s).append(" ");
//            tv_feature_value.setText(builder.toString());
//        }



        tv_feature_value.setText(builder.toString());
    }

    private void histogram_gray(){
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
//        Log.d("JUG", "Execute");
//        FeatureDetector detector;
//        DescriptorExtractor descriptorExtractor;
//        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
//        Mat descriptors1 = new Mat();;
//
//
//        Log.d("JUG", "Before Extract Feature");
//        detector = FeatureDetector.create(FeatureDetector.SIFT);
//        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
//
//
//        Log.d("JUG", "Detect Keypoints");
//        detector.detect(sampledImgMat, keypoints1);
//
//        keypointsObject1 = keypoints1.toArray().length;
//
//        descriptorExtractor.compute(sampledImgMat,keypoints1,descriptors1);
//
//        Mat src1 = new Mat(imageBitmap.getHeight(), imageBitmap.getWidth(), CvType.CV_8UC4);
//        Bitmap image1 = Bitmap.createBitmap(src1.cols(), src1.rows(), Bitmap.Config.ARGB_8888);
//        return image1;
//
//        //Bitmap image1 = Bitmap.createBitmap()

//        ORB orb = ORB.create();
//        MatOfKeyPoint keypoints = new MatOfKeyPoint();
//        Mat descriptors = new Mat();
//        orb.detectAndCompute(sampledImgMat, new Mat(), keypoints, descriptors)

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);



    }
}
