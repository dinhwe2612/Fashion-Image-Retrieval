package com.example.fashion_image_retrieval;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.RelativeLayout;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ResultAdapter extends RecyclerView.Adapter<ResultAdapter.ResultViewHolder> {
    List<String> images;
    List<String> relevant = new ArrayList<>();
    boolean isFeedback = false;
    List<Boolean> feedBacks;

    public ResultAdapter(List<String> images) {
        this.images = images;
    }

    public void setIsFeedback(boolean isFeedback) {
        this.isFeedback = isFeedback;
        feedBacks = new ArrayList<>(Collections.nCopies(images.size(), false));
    }

    public List<String> getRelevantImages() {
        return relevant;
    }

    @NonNull
    @Override
    public ResultViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.image_item, parent, false);
        return new ResultViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ResultViewHolder holder, int position) {
        Log.d("in", images.get(position));
        Glide.with(holder.imageView.getContext()).load(images.get(position)).into(holder.imageView);
        if (isFeedback) {
            holder.check.setVisibility(View.VISIBLE);
        } else {
            holder.check.setVisibility(View.GONE);
        }
    }

    @Override
    public int getItemCount() {
        return images.size();
    }

    public static class ResultViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;
        RelativeLayout check;
        public ResultViewHolder(@NonNull View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.image_view);
            check = itemView.findViewById(R.id.check);
        }
    }
}
