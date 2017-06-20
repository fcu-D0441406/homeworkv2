package com.example.n.myapplication;

import android.app.Dialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Random;

public class MainActivity extends AppCompatActivity {
    private Button start_game,choose_music,backpack;
    public static int  [] music_list_id = {R.raw.dudulu,R.raw.babu,R.raw.music1,R.raw.music2};
    public static String [] music_name = {"dudulu","babu","music1","music2"};
    public static  int music_id = music_list_id[3];
    public static int longtime_buff=5,half_buff=5;
    public static int all_point = 0;
    private TextView textView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        save_load();
        Main2Activity.re_number = 0;
        Main2Activity.number = 0;
        get_id();
    }

    private void get_id(){
        choose_music = (Button) findViewById(R.id.choose_music);
        choose_music.setOnClickListener(listen);
        backpack = (Button) findViewById(R.id.backpack);
        backpack.setOnClickListener(listen);
        textView = (TextView)findViewById(R.id.textView);
        textView.setText(Integer.toString(all_point));
        start_game = (Button) findViewById(R.id.start);
        start_game.setOnClickListener(listen);
    }

    private void save_load(){
        SharedPreferences sharedPreferences = getSharedPreferences("game_result",0);
        //sharedPreferences.edit().putInt("point",all_point).putInt("longtime_buff",longtime_buff).putInt("half_buff",half_buff).commit();
        all_point = sharedPreferences.getInt("all_point",0);
        longtime_buff = sharedPreferences.getInt("longtime_buff",5);
        half_buff = sharedPreferences.getInt("half_buff",5);
    }
    private Dialog dialog_music,dialog_backpack;
    private View.OnClickListener listen = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if(v.getId()==R.id.start){
                Intent it = new Intent();
                it.setClass(MainActivity.this,Main2Activity.class);
                startActivity(it);
            }
            else if(v.getId()==R.id.choose_music){
                dialog_music = new Dialog(MainActivity.this);
                dialog_music.setCancelable(true);
                dialog_music.setContentView(R.layout.set_music);
                exit_music = (Button) dialog_music.findViewById(R.id.exit);
                exit_music.setOnClickListener(listen_exit);
                dialog_music.show();
                ListView listView = (ListView) dialog_music.findViewById(R.id.choose_music);
                ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(MainActivity.this,R.array.music_name,android.R.layout.simple_list_item_1);
                listView.setAdapter(adapter);
                listView.setOnItemClickListener(listener);
            }
            else if(v.getId()==R.id.backpack){
                dialog_backpack = new Dialog(MainActivity.this);
                dialog_backpack.setCancelable(true);

            }
        }
    };
    private Button exit_music;
    private AdapterView.OnItemClickListener listener = new AdapterView.OnItemClickListener(){
        public void onItemClick(AdapterView<?> parent,View view,int position,long id){
            music_id = music_list_id[position];
            Toast.makeText(MainActivity.this,"音樂更換成功",Toast.LENGTH_LONG).show();
        }
    };
    private View.OnClickListener listen_exit = new View.OnClickListener(){
        public void onClick(View v) {
            dialog_music.cancel();
        }
    };
}

