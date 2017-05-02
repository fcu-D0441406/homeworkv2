package com.example.n.myapplication;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Timer;
import java.util.TimerTask;

public class Main3Activity extends AppCompatActivity {
    private Button back_menu;
    private int all_point2,half_buff2,longtime_buff2;
    private int a = 1;
    private TextView result;
    Timer timer = new Timer();
    private int [] picture = new int[]{R.drawable.back_one,R.drawable.back_two,R.drawable.back_three,R.drawable.back_four,R.drawable.back_five,R.drawable.back_six,R.drawable.back_seven,R.drawable.back_eight,R.drawable.back_night};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main3);
        result = (TextView)findViewById(R.id.textView2);
        back_menu = (Button) findViewById(R.id.back_menu);
        back_menu.setOnClickListener(listen);
        Intent it2 = getIntent();
        SharedPreferences sharedPreferences = getSharedPreferences("game_result",0);
        all_point2 = sharedPreferences.getInt("all_point",0);
        result.setText("總共得到"+Integer.toString(it2.getExtras().getInt("point"))+"分");
        if(it2.getExtras().getInt("point")>=5){
            result.setBackgroundResource(R.drawable.ranking_a);
        }
        else{
            result.setBackgroundResource(R.drawable.ranking_b);
        }
        all_point2 = all_point2+it2.getExtras().getInt("point");
        half_buff2 = it2.getExtras().getInt("h");
        longtime_buff2 = it2.getExtras().getInt("l");
        timer.schedule(timertask,0,100);
    }
    private View.OnClickListener listen = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            Intent it2 = getIntent();
            SharedPreferences sharedPreferences = getSharedPreferences("game_result",0);
            sharedPreferences.edit().putInt("all_point",all_point2).putInt("half_buff",half_buff2).putInt("longtime_buff",longtime_buff2).commit();
            Intent it = new Intent();
            it.setClass(Main3Activity.this,MainActivity.class);
            startActivity(it);
        }
    };
    public void onBackPressed() {

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
            if(msg.what==1) {
                back_menu.setBackgroundResource(picture[a]);
                a++;
                if(a==8){
                    a=0;
                }
            }
        }
    };
}
