package com.example.project_a;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
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
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
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
    ImageView imv_gallery1, imv_gallery2, imv_feature_image;

    Mat sampledImgMat, sampledImgMat1, sampledImgMat2;
    Bitmap imageBitmap, imageBitmap1, imageBitmap2, grayBitmap1, grayBitmap2, mathchBitmapORB;
    Boolean src1selected, src2selected;

    Uri imageUri;
    private int keypointsObject1, keypointsObject2, keypointMatches;
    private final int MAX_MATCHES = 50;

    float[] histogram_img1, histogram_img2;

    private int REQUEST_CODE_GALLERRY = 100;

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
        btn_orb = findViewById(R.id.btn_orb);

        imv_gallery1 = findViewById(R.id.imv_gallery1);
        imv_gallery2 = findViewById(R.id.imv_gallery2);
        imv_feature_image = findViewById(R.id.imv_feature_image);

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
                histogram_img1 = get_histogram_rgb(sampledImgMat1);
                Log.d(TAG, histogram_img1.toString());
                try{
                    histogram_img2 = get_histogram_rgb(sampledImgMat2);
                    Log.d(TAG, histogram_img2.toString());
                }catch (NullPointerException e){
                    e.printStackTrace();
                }
            }
        });
        btn_orb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mathchBitmapORB = matching_images_ORB(sampledImgMat1, sampledImgMat2);
                imv_feature_image.setImageBitmap(mathchBitmapORB);
            }
        });



    }
//
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

    private Bitmap matching_images_ORB(Mat sampledImgMat1, Mat sampledImgMat2){

        FeatureDetector detector;
        MatOfKeyPoint keypoints1, keypoints2;
        DescriptorExtractor descriptorExtractor;
        Mat descriptors1, descriptors2;
        DescriptorMatcher descriptorMatcher;
        MatOfDMatch matches = new MatOfDMatch();
        keypoints1 = new MatOfKeyPoint();
        keypoints2 = new MatOfKeyPoint();
        descriptors1 = new Mat();
        descriptors2 = new Mat();

        detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        //Get keypoints of 2 images
        detector.detect(sampledImgMat1, keypoints1);
        detector.detect(sampledImgMat2, keypoints2);

        keypointsObject1 = keypoints1.toArray().length;
        keypointsObject2 = keypoints2.toArray().length;

        //Get descriptorห
        descriptorExtractor.compute(sampledImgMat1, keypoints1, descriptors1);
        descriptorExtractor.compute(sampledImgMat2, keypoints2, descriptors2);

        //Matching 2 descriptorห
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

        float []distance = new float[matches.toList().size()];
        float sum_distance = 0.0f;

        for (int i=0; i < matches.toList().size(); i++){
            //distance[i] = listOfDMatch.get(i).distance;
            distance[i] = matches.toList().get(i).distance;
            sum_distance = sum_distance + distance[i];
        }
        Log.d(TAG, String.valueOf(sum_distance));

        //TODO: SORT DISANCEd

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
        if ( imageOnly){
            MatOfDMatch emptyMatch = new MatOfDMatch();
            MatOfKeyPoint emptyKey1 = new MatOfKeyPoint();
            MatOfKeyPoint emptyKey2 = new MatOfKeyPoint();
            Features2d.drawMatches(im1, emptyKey1, im2, emptyKey2, emptyMatch, out);
        } else {
            Features2d.drawMatches(im1, key1, im2, key2, matches, out);
        }
        Bitmap bmp = Bitmap.createBitmap(out.cols(), out.rows(), Bitmap.Config.ARGB_8888);
        Imgproc.cvtColor(out, out, Imgproc.COLOR_BGR2RGB);
        //Core.putText(out, "FRAME", new Point(img1.width() / 2,30), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(0,255,255),3);
        //Core.putText(out, "MATCHED", new Point(img1.width() + img2.width() / 2,30), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(255,0,0),3);
        return out;
    }

}


