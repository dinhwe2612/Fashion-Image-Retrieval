package com.example.fashion_image_retrieval;

import android.widget.ImageView;

import androidx.databinding.BindingAdapter;

import com.bumptech.glide.Glide;

public final class Utils {
    private Utils() {}
    @BindingAdapter("imageUrl")
    public static void setImageResource(ImageView imageView, String imageUrl) {
        Glide.with(imageView.getContext()).load(imageUrl).into(imageView);
    }
}
