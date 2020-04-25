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
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class CompareImages extends AppCompatActivity {
    private static final String TAG = "JUG";

    Button btn_gallery1, btn_gallery2, btn_histogram_rgb, btn_histogram_gray, btn_brisk, btn_orb;
    ImageView imv_gallery1, imv_gallery2;

    Mat sampledImgMat, sampledImgMat1, sampledImgMat2;
    Bitmap imageBitmap, imageBitmap1, imageBitmap2, grayBitmap, contourBitmap;
    Boolean src1selected, src2selected;

    Uri imageUri;

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

        imv_gallery1 = findViewById(R.id.imv_gallery1);
        imv_gallery2 = findViewById(R.id.imv_gallery2);

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





}
