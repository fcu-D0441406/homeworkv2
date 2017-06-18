package com.example.n.myapplication;


import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;


/**
 * A simple {@link Fragment} subclass.
 */
public class BlankFragment extends Fragment {
    public static String [] topic_all = new String[50];
    public static TextView topic;
    public BlankFragment() {
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        return inflater.inflate(R.layout.fragment_blank, container, false);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        set_topic();
        topic = (TextView) getActivity().findViewById(R.id.topic);
        show_topic(Main2Activity.number);
    }
    public void set_topic(){
        topic_all[0] = "_門一腳 ";
        topic_all[1] = "_頭蛇尾";
        topic_all[2] = "美國2017年總統大選當選者為:";
        topic_all[3] = "三國演義作者為?";
    }
    public static void show_topic(int number){
        topic.setText(topic_all[number]);
    }
}