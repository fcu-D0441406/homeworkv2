package com.example.n.myapplication;


import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;


/**
 * A simple {@link Fragment} subclass.
 */
public class BlankFragment3 extends Fragment {


    public BlankFragment3() {
        // Required empty public constructor
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_blank_fragment3, container, false);
    }
    public static TextView double_count,point_count,hidden_count;
    public static Button add_dt,add_hd;
    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        double_count = (TextView) getActivity().findViewById(R.id.double_count);
        point_count = (TextView) getActivity().findViewById(R.id.point_count);
        hidden_count = (TextView) getActivity().findViewById(R.id.hidden_count);
        add_dt = (Button) getActivity().findViewById(R.id.buy_doubletime);
        add_hd = (Button) getActivity().findViewById(R.id.buy_halftime);
        BlankFragment3.hidden_count.setText(Integer.toString(MainActivity.half_buff));
        BlankFragment3.double_count.setText(Integer.toString(MainActivity.longtime_buff));
        BlankFragment3.point_count.setText(Integer.toString(MainActivity.all_point));
        add_dt.setOnClickListener(listen);
        add_hd.setOnClickListener(listen);
    }
    private View.OnClickListener listen = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            switch(v.getId()){
                case R.id.buy_doubletime:
                    if(MainActivity.all_point>0){
                        MainActivity.all_point--;
                        MainActivity.longtime_buff++;
                    }
                    else{
                        Toast.makeText(getActivity(),"金錢不足",Toast.LENGTH_LONG).show();
                    }
                    break;
                case R.id.buy_halftime:
                    if(MainActivity.all_point>0){
                        MainActivity.all_point--;
                        MainActivity.half_buff++;
                    }
                    else{
                        Toast.makeText(getActivity(),"金錢不足",Toast.LENGTH_LONG).show();
                    }
                    break;
            }
            BlankFragment3.hidden_count.setText(Integer.toString(MainActivity.half_buff));
            BlankFragment3.double_count.setText(Integer.toString(MainActivity.longtime_buff));
            BlankFragment3.point_count.setText(Integer.toString(MainActivity.all_point));
        }
    };

}
