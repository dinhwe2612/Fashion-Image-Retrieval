package com.example.fashion_image_retrieval;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.view.Menu;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;

import com.bumptech.glide.Glide;
import com.google.android.material.navigation.NavigationView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;

import com.example.fashion_image_retrieval.databinding.ActivityMainBinding;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    ResultAdapter adapter;
    String BaseURL = "http://10.0.89.146:8000";
    Uri imageUriQuery;
    String dataset = "FashionIQ";
    String topK = "10";
    String text = "";
    private static final int REQUEST_PERMISSIONS_CODE = 101;

    private static final int IMAGE_PICKER_REQUEST = 1;

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICKER_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == IMAGE_PICKER_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            if (imageUri != null) {
                imageUriQuery = imageUri;
                Glide.with(this)
                        .load(imageUri)
                        .into(binding.queryImage);
            }
        }
    }
    private File uriToFile(Uri uri) {
        if (uri == null) return null;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                String fileName = "temp_image.jpg"; // Create a temporary file
                File tempFile = new File(getExternalFilesDir(null), fileName);
                try (InputStream inputStream = getContentResolver().openInputStream(uri);
                     FileOutputStream outputStream = new FileOutputStream(tempFile)) {
                    byte[] buffer = new byte[1024];
                    int bytesRead;
                    while ((bytesRead = inputStream.read(buffer)) != -1) {
                        outputStream.write(buffer, 0, bytesRead);
                    }
                }
                return tempFile;
            } catch (IOException e) {
                Log.e("URI_TO_FILE", "Failed to create file from URI", e);
            }
        } else {
            String[] projection = {MediaStore.Images.Media.DATA};
            Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
            if (cursor != null) {
                int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
                cursor.moveToFirst();
                String filePath = cursor.getString(columnIndex);
                cursor.close();
                return new File(filePath);
            }
        }
        return null;
    }

    void predict() {
        File imageQuery = uriToFile(imageUriQuery); // Convert Uri to File
        OkHttpClient client = new OkHttpClient();
        MediaType MEDIA_TYPE_IMAGE = MediaType.parse("image/*");

        // Create a multipart body builder
        MultipartBody.Builder bodyBuilder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("dataset", dataset) // Add text field
                .addFormDataPart("text", text);

        if (imageQuery != null) {
            bodyBuilder.addFormDataPart("file", imageQuery.getName(),
                    RequestBody.create(imageQuery, MEDIA_TYPE_IMAGE));
        }

        Log.d("REQUEST_URL", BaseURL + "/predict/");
        Log.d("REQUEST_FIELDS", "dataset=" + dataset + ", text=" + text + ", k=" + topK);

        // Build the request body
        RequestBody requestBody = bodyBuilder.build();

        // Create the request
        Request request = new Request.Builder()
                .url(BaseURL + "/predict/")
                .addHeader("Content-Type", "multipart/form-data")
                .post(requestBody)
                .build();

        // Send the request
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                Log.e("UPLOAD_ERROR", "Failed to upload", e);
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    String jsonResponse = response.body().string();

                    // Parse the JSON response and update the RecyclerView
                    List<String> imageNames = parseJsonResponse(jsonResponse);
                    runOnUiThread(() -> {
                        adapter = new ResultAdapter(imageNames);
                        binding.rcvResult.setAdapter(adapter);
                    });
                } else {
                    Log.e("UPLOAD_ERROR", "Server error: " + response);
                }
            }
        });
    }

    private List<String> parseJsonResponse(String jsonResponse) {
        List<String> imageNames = new ArrayList<>(); // Initialize the list to store image names
        try {
            JSONObject jsonObject = new JSONObject(jsonResponse);
            JSONArray topImagesArray = jsonObject.getJSONArray("top_images");
            for (int i = 0; i < topImagesArray.length(); i++) {
                JSONArray imageInfo = topImagesArray.getJSONArray(i);
                String imageName = BaseURL + "/images/" + dataset + "/" + imageInfo.getString(0);
                imageNames.add(imageName);
            }
        } catch (Exception e) {
            Log.e("JSON_PARSE_ERROR", "Failed to parse JSON", e);
        }
        return imageNames;
    }

    @SuppressLint("ObsoleteSdkInt")
    private void checkAndRequestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
                    checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
                    checkSelfPermission(Manifest.permission.MANAGE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

                // Request permissions
                requestPermissions(
                        new String[]{
                                Manifest.permission.READ_EXTERNAL_STORAGE,
                                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                                Manifest.permission.CAMERA,
                                Manifest.permission.MANAGE_EXTERNAL_STORAGE
                        },
                        REQUEST_PERMISSIONS_CODE
                );
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSIONS_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (!allGranted) {
                // Show message to the user explaining why permissions are necessary
                Toast.makeText(this, "Permissions are required to proceed.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        checkAndRequestPermissions();

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        DrawerLayout drawer = binding.drawerLayout;
        NavigationView navigationView = binding.navView;
        View headerView = navigationView.getHeaderView(0);
        ImageButton backButton = headerView.findViewById(R.id.back_button);
        EditText ipEdit = headerView.findViewById(R.id.ip_edit);
        Button ipEnterButton = headerView.findViewById(R.id.enter_ip_button);
        ipEnterButton.setOnClickListener(v -> {
            BaseURL = "http://" + ipEdit.getText().toString().trim() + ":8000";
            Toast.makeText(this, "IP address set to: " + BaseURL, Toast.LENGTH_SHORT).show();
        });
        backButton.setOnClickListener(v->drawer.closeDrawer(binding.navView));
        binding.burgerButton.setOnClickListener(v->drawer.openDrawer(binding.navView));
        binding.searchButton.setOnClickListener(v -> {
            text = binding.searchEdit.getText().toString().trim();
            topK = "10";
            predict();
            Toast.makeText(this, "Searching for: " + text, Toast.LENGTH_SHORT).show();
        });
        binding.imagePicker.setOnClickListener(v->{
            openImagePicker();
        });
        List<String> images = new ArrayList<>();
        adapter = new ResultAdapter(images);
        binding.rcvResult.setAdapter(adapter);
        binding.rcvResult.setLayoutManager(new GridLayoutManager(this, 2));
    }
}