package com.example.n.myapplication;

import android.os.Build;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.w3c.dom.Text;

import java.util.Timer;
import java.util.TimerTask;

import static java.lang.Thread.sleep;

public class Main2Activity extends AppCompatActivity {
    BlankFragment b1 = new BlankFragment();
    BlankFragment2 b2 = new BlankFragment2();
    private int [] count_image = new int []{R.drawable.zero,R.drawable.one,R.drawable.two,R.drawable.three,R.drawable.four,R.drawable.five,R.drawable.six};
    public static int number = 0,re_number = 0;
    public static int [] answer_number = {0,0};
    public static TextView recip_text;
    public static  int point = 0;
    public static int recip = 6;
    public static Timer timer = new Timer();
    public static TextView point_textview;
    public static Button double_time,hidden;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
        timer.schedule(timertask,0,1000);
        get_id();
    }

    private void get_id(){
        recip_text = (TextView) findViewById(R.id.recip_text);
        point_textview = (TextView)findViewById(R.id.point);
        double_time = (Button) findViewById(R.id.double_time);
        hidden = (Button) findViewById(R.id.hidden);
        double_time.setOnClickListener(buff);
        hidden.setOnClickListener(buff);
    }

    //返回鍵取消
    public void onBackPressed() {
    }
    //
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
            recip--;
            if(msg.what==1) {
                if(recip!=0){
                    recip_text.setBackgroundResource(count_image[recip-1]);
                }
                /*else if(recip==0){
                    delay();
                }*/
            }
        }
    };
    public static void lock_button(){
        BlankFragment2.answer_aa.setEnabled(false);
        BlankFragment2.answer_bb.setEnabled(false);
        BlankFragment2.answer_cc.setEnabled(false);
        BlankFragment2.answer_dd.setEnabled(false);
    }
    public static void open_button(){
        BlankFragment2.answer_aa.setEnabled(true);
        BlankFragment2.answer_bb.setEnabled(true);
        BlankFragment2.answer_cc.setEnabled(true);
        BlankFragment2.answer_dd.setEnabled(true);
        double_time.setEnabled(true);
        hidden.setEnabled(true);
        double_time.setBackgroundResource(R.drawable.double_times);
        hidden.setBackgroundResource(R.drawable.hidden);
    }
    private View.OnClickListener buff = new View.OnClickListener() {
        @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN)
        @Override
        public void onClick(View v) {
            if(v.getId()==R.id.double_time){
                if(MainActivity.longtime_buff>0){
                    recip+=3;
                    MainActivity.longtime_buff--;
                    double_time.setEnabled(false);
                    double_time.setBackgroundResource(R.drawable.false1);
                }
                else{
                    Toast.makeText(Main2Activity.this,"你的道具不足",Toast.LENGTH_LONG).show();
                }
            }
            if(v.getId()==R.id.hidden){
                if(MainActivity.half_buff>0){
                    MainActivity.half_buff--;
                    hidden.setEnabled(false);
                    hidden.setBackgroundResource(R.drawable.false1);
                    delete_answer();
                }
            }
            //Toast.makeText(Main2Activity.this,Integer.toString(MainActivity.longtime_buff)+" "+Integer.toString(MainActivity.half_buff),Toast.LENGTH_LONG).show();
        }
    };
    private void delete_answer(){
        BlankFragment2.answer[0] = (Button) findViewById(R.id.answer_a);
        BlankFragment2.answer[1] = (Button) findViewById(R.id.answer_b);
        BlankFragment2.answer[2] = (Button) findViewById(R.id.answer_c);
        BlankFragment2.answer[3] = (Button) findViewById(R.id.answer_d);
        int  [] answer = new int [3];
        int a = 0;
        for(int i = 0;i<4;i++){
            if(i!=answer_number[number]){
                answer[a] = i;
                a++;
            }
        }
        for(int i = 0;i<3;i++){
            int temp = (int)(Math.random()*3);
            int temp2 = (int)(Math.random()*3);
            int temp3;
            temp3 = answer[temp];
            answer[temp] = answer[temp2];
            answer[temp2] = temp3;
        }
        for(int i = 0;i<2;i++){
            BlankFragment2.answer[answer[i]].setEnabled(false);
        }
    }
    public static void delay(){
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
