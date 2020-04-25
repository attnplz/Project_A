package com.example.project_a;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    Button btn_load, btn_preprocessing, btn_feature_extraction, btn_prediction, btn_capture, btn_compare;
    ImageView imv;
    Bitmap bitmap;

    private final int CAMERA_RESULT_CODE = 100;
    private final int LOADING_RESULT_CODE = 120;
    private Uri mImgeFileUri;

    OutputStream outputStream;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn_capture = findViewById(R.id.btn_capture);
        btn_load = findViewById(R.id.btn_load);
        btn_preprocessing = findViewById(R.id.btn_preprocessing);
        btn_feature_extraction = findViewById(R.id.btn_feature_extraction);
        btn_prediction = findViewById(R.id.btn_prediction);
        btn_compare = findViewById(R.id.btn_compare);
        imv = findViewById(R.id.imv);

        btn_capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, CAMERA_RESULT_CODE);
            }
        });

        btn_load.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, LOADING_RESULT_CODE);
            }
        });

        btn_preprocessing.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, PreprocessingActivity.class);
                startActivity(intent);
            }
        });

        btn_feature_extraction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, FeatureExtractionActivity.class);
                startActivity(intent);
            }
        });

        btn_prediction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, PredictionActivity.class);
                startActivity(intent);
            }
        });

        btn_compare.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, CompareImages.class);
                startActivity(intent);
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permission, @NonNull int[] grantResults){
        if(requestCode == CAMERA_RESULT_CODE){
            if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED){
                Intent it = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                if(it.resolveActivity(getPackageManager()) != null){
                    startActivityForResult(it,CAMERA_RESULT_CODE);
                }
            }else{
                Toast.makeText(MainActivity.this,"Cannot use camera", Toast.LENGTH_LONG).show();
            }
        }else{
            super.onRequestPermissionsResult(requestCode, permission, grantResults);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == CAMERA_RESULT_CODE) {
                Bundle bd = data.getExtras();
                Bitmap bmp = (Bitmap) bd.get("data");
                imv.setImageBitmap(bmp);
                String fileName = createImageFile();
                storeImageToInternal(fileName);
            }
            if (requestCode == LOADING_RESULT_CODE){
                Uri imgUri = data.getData();
                try{
                    InputStream inputStream = getContentResolver().openInputStream(imgUri);
                    bitmap = BitmapFactory.decodeStream(inputStream);
                    imv.setImageBitmap(bitmap);
                }catch (FileNotFoundException e){
                    e.printStackTrace();
                }
            }
        }
    }

    private void storeImageToInternal(String fileName){

        //Store image into external storage
        BitmapDrawable drawable = (BitmapDrawable) imv.getDrawable();
        Bitmap bitmap = drawable.getBitmap();

        try{
            //Get a public path on the device storage for saving the file
            File path = Environment.getExternalStorageDirectory();

            //Create directory for saving the image
            File saveDir = new File(path.getAbsolutePath() + "/Demo");

            //If the directory is not existed
            if(!saveDir.exists()){
                saveDir.mkdir();
            }

            //Create the image file within the directory
            File fileDir = new File(saveDir, fileName);     //Create the file

            Toast.makeText(getApplicationContext(),"Image Save to External Storage",Toast.LENGTH_SHORT).show();

            //Write into the image file by the Bitmap content
            outputStream = new FileOutputStream(fileDir);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);

            MediaScannerConnection.scanFile(this.getApplicationContext(),
                    new String[]{fileDir.toString()}, null,
                    new MediaScannerConnection.OnScanCompletedListener() {
                        @Override
                        public void onScanCompleted(String path, Uri uri) {
                        }
                    });

        }catch(Exception e){
            e.printStackTrace();
        }
    }

    private String createImageFile(){
        //Create an image file name
        SimpleDateFormat formatter = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US);
        Date now = new Date();
        String fileName = formatter.format(now) + ".jpg" ;

//        File storageDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM) + "/Camera/");
//        if (!storageDir.exists()) storageDir.mkdir();
//        File image = File.createTempFile(timeStamp,".jpg", storageDir);
//        return image;

        return fileName;
    }
}
