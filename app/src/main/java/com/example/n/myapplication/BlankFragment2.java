package com.example.n.myapplication;


import android.content.Intent;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Timer;
import java.util.TimerTask;

import static com.example.n.myapplication.MainActivity.*;


/**
 * A simple {@link Fragment} subclass.
 */
public class BlankFragment2 extends Fragment {
    private String [] answer_a = new String[50];
    private String [] answer_b = new String[50];
    private String [] answer_c = new String[50];
    private String [] answer_d = new String[50];
    public static Button answer_aa,answer_bb,answer_cc,answer_dd;
    public static Button [] answer = {answer_aa,answer_bb,answer_cc,answer_dd};
    public int [] aaaaa = new int[10];
    private Timer timer = new Timer();
    private MediaPlayer mediaPlayer = new MediaPlayer();
    public BlankFragment2() {
        // Required empty public constructor
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_blank_fragment2, container, false);
    }
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        set_topic();
        answer_aa = (Button) getActivity().findViewById(R.id.answer_a);
        answer_bb = (Button) getActivity().findViewById(R.id.answer_b);
        answer_cc = (Button) getActivity().findViewById(R.id.answer_c);
        answer_dd = (Button) getActivity().findViewById(R.id.answer_d);
        answer_aa.setOnClickListener(listen);
        answer_bb.setOnClickListener(listen);
        answer_cc.setOnClickListener(listen);
        answer_dd.setOnClickListener(listen);
        show_topic(Main2Activity.number);
        timer.schedule(timertask,0,1000);
    }
    public void set_topic(){
        answer_a[0] = "臨";
        answer_b[0] = "霖";
        answer_c[0] = "林";
        answer_d[0] = "玲";
        answer_a[1] = "虎";
        answer_b[1] = "唬";
        answer_c[1] = "汻";
        answer_d[1] = "萀";

        answer_a[2] = "Donald Trump";
        answer_b[2] = "Michael Jackson";
        answer_c[2] = "Pig";
        answer_d[2] = "Barack Obama";

        answer_a[3] = "羅貫中";
        answer_b[3] = "陳壽";
        answer_c[3] = "施耐庵";
        answer_d[3] = "吳承恩";
    }
    int random;
    public static int [] button_id = {R.id.answer_a,R.id.answer_b,R.id.answer_c,R.id.answer_d};
    public void show_topic(int number){
        //Toast.makeText(getActivity(),Integer.toString(Main2Activity.re_number),Toast.LENGTH_LONG).show();
        if(Main2Activity.re_number==4){
            show_end();
        }
        random = (int)(Math.random()*3+1);
        Main2Activity.answer_number[Main2Activity.re_number]=random;
        System.out.println(random);

        Main2Activity.re_number++;
        if(random==1){
            answer_aa.setText(answer_b[number]);
            answer_bb.setText(answer_a[number]);
            answer_cc.setText(answer_c[number]);
            answer_dd.setText(answer_d[number]);
        }
        else if(random==2){
            answer_aa.setText(answer_c[number]);
            answer_bb.setText(answer_b[number]);
            answer_cc.setText(answer_a[number]);
            answer_dd.setText(answer_d[number]);
        }
        else if(random==3){
            answer_aa.setText(answer_d[number]);
            answer_bb.setText(answer_b[number]);
            answer_cc.setText(answer_c[number]);
            answer_dd.setText(answer_a[number]);
        }
        /*answer_aa.setText(answer_a[number]);
                 answer_bb.setText(answer_b[number]);
                answer_cc.setText(answer_c[number]);
                answer_dd.setText(answer_d[number]);*/
    }
    private View.OnClickListener listen = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if(Main2Activity.number==0){
                if(v.getId()==button_id[random]){
                    Toast.makeText(getActivity(),"答對了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                    add_point();
                    play_music();
                }
                else{
                    Toast.makeText(getActivity(),"答錯了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                }
            }
            else if(Main2Activity.number==1){
                if(v.getId()==button_id[random]){
                    Toast.makeText(getActivity(),"答對了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                    add_point();
                    play_music();
                }
                else{
                    Toast.makeText(getActivity(),"答錯了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                }
            }
            if(Main2Activity.number==2){
                if(v.getId()==button_id[random]){
                    Toast.makeText(getActivity(),"答對了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                    add_point();
                    play_music();
                }
                else{
                    Toast.makeText(getActivity(),"答錯了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                }
            }
            if(Main2Activity.number==3){
                if(v.getId()==button_id[random]){
                    Toast.makeText(getActivity(),"答對了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                    add_point();
                    play_music();
                }
                else{
                    Toast.makeText(getActivity(),"答錯了",Toast.LENGTH_LONG).show();
                    Main2Activity.lock_button();
                }
            }
        }
    };
    private void add_point(){
        Main2Activity.point+=Main2Activity.recip-1;
        Main2Activity.point_textview.setText("目前分數 "+Integer.toString(Main2Activity.point));
    }
    private  void change_topic(){
        Main2Activity.delay();
        Main2Activity.recip = 6;
        Main2Activity.number = Main2Activity.number+1;
        show_topic(Main2Activity.number);
        BlankFragment.show_topic(Main2Activity.number);
        Main2Activity.open_button();
    }
    public  void show_end(){
        Intent it = new Intent();
        it.setClass(getActivity(),Main3Activity.class);
        it.putExtra("point",Main2Activity.point);
        it.putExtra("l",MainActivity.longtime_buff);
        it.putExtra("h",MainActivity.half_buff);
        startActivity(it);;
        System.exit(0);
    }
    TimerTask timertask = new TimerTask() {
        @Override
        public void run() {
            Message msg = new Message();
            msg.what = 1;
            handler.sendMessage(msg);
        }
    };
    Handler handler = new Handler(){
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            if(msg.what==1){
                if(Main2Activity.recip==0){
                    change_topic();
                }
            }
        }
    };
    private void play_music(){
        mediaPlayer = MediaPlayer.create(getActivity(),MainActivity.music_id);
        mediaPlayer.start();
    }
}