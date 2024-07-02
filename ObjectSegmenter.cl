/*
OpenCL RandomForestClassifier
classifier_class_name = ObjectSegmenter
feature_specification = gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 sobel_of_gaussian_blur=1
num_ground_truth_dimensions = 2
num_classes = 2
num_features = 4
max_depth = 2
num_trees = 100
feature_importances = 0.12547246285783717,0.45127598099087446,0.4160511581279878,0.007200398023300627
positive_class_identifier = 2
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i0<1484.182861328125){
 s0+=1.0;
} else {
 if(i2<2.02484130859375){
  s0+=1.0;
 } else {
  s0+=0.008;
  s1+=0.992;
 }
}
if(i0<1532.4112548828125){
 if(i3<4328.71484375){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i2<68.091064453125){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
}
if(i1<-1.79766845703125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<-3.088134765625){
 s0+=1.0;
} else {
 if(i1<33.90179443359375){
  s0+=0.05555555555555555;
  s1+=0.9444444444444444;
 } else {
  s1+=1.0;
 }
}
if(i0<1478.29443359375){
 if(i3<4480.8544921875){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i1<-3.088134765625){
  s0+=1.0;
 } else {
  s0+=0.009615384615384616;
  s1+=0.9903846153846154;
 }
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.05555555555555555;
  s1+=0.9444444444444444;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 if(i2<226.35107421875){
  s0+=0.25;
  s1+=0.75;
 } else {
  s0+=0.008849557522123894;
  s1+=0.9911504424778761;
 }
}
if(i2<2.02484130859375){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.25;
  s1+=0.75;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.4;
  s1+=0.6;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.2;
  s1+=0.8;
 } else {
  s0+=0.008771929824561403;
  s1+=0.9912280701754386;
 }
}
if(i0<1662.4263916015625){
 if(i2<-59.567535400390625){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i0<2049.42333984375){
  s0+=0.08108108108108109;
  s1+=0.918918918918919;
 } else {
  s1+=1.0;
 }
}
if(i2<20.33642578125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1453.120849609375){
 s0+=1.0;
} else {
 if(i1<-2.8719482421875){
  s0+=1.0;
 } else {
  s0+=0.016129032258064516;
  s1+=0.9838709677419355;
 }
}
if(i1<-5.7362060546875){
 s0+=1.0;
} else {
 if(i2<226.35107421875){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s0+=0.009174311926605505;
  s1+=0.9908256880733946;
 }
}
if(i2<5.2626953125){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.16666666666666666;
  s1+=0.8333333333333334;
 } else {
  s1+=1.0;
 }
}
if(i1<10.61090087890625){
 if(i1<-2.8719482421875){
  s0+=1.0;
 } else {
  s0+=0.5;
  s1+=0.5;
 }
} else {
 s1+=1.0;
}
if(i1<-3.088134765625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<12.61688232421875){
 if(i3<4560.5205078125){
  s0+=0.9841269841269841;
  s1+=0.015873015873015872;
 } else {
  s1+=1.0;
 }
} else {
 s1+=1.0;
}
if(i1<-5.7362060546875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-5.44403076171875){
 s0+=1.0;
} else {
 if(i2<509.73468017578125){
  s0+=0.04;
  s1+=0.96;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i2<226.35107421875){
  s0+=0.25;
  s1+=0.75;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i2<224.71023559570312){
  s0+=0.2222222222222222;
  s1+=0.7777777777777778;
 } else {
  s1+=1.0;
 }
}
if(i1<-5.49468994140625){
 s0+=1.0;
} else {
 if(i0<1966.765625){
  s0+=0.03571428571428571;
  s1+=0.9642857142857143;
 } else {
  s1+=1.0;
 }
}
if(i0<1760.7919921875){
 if(i3<4003.890625){
  s0+=1.0;
 } else {
  s0+=0.16666666666666666;
  s1+=0.8333333333333334;
 }
} else {
 if(i0<1945.0845947265625){
  s0+=0.2222222222222222;
  s1+=0.7777777777777778;
 } else {
  s1+=1.0;
 }
}
if(i1<-1.79766845703125){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.5;
  s1+=0.5;
 } else {
  s0+=0.019801980198019802;
  s1+=0.9801980198019802;
 }
}
if(i1<3.32147216796875){
 s0+=1.0;
} else {
 if(i1<35.67340087890625){
  s0+=0.07142857142857142;
  s1+=0.9285714285714286;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i1<31.94683837890625){
  s0+=0.2;
  s1+=0.8;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i1<31.94683837890625){
  s0+=0.14285714285714285;
  s1+=0.8571428571428571;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i0<2048.297119140625){
  s0+=0.05128205128205128;
  s1+=0.9487179487179487;
 } else {
  s1+=1.0;
 }
}
if(i2<5.2626953125){
 s0+=1.0;
} else {
 if(i0<1967.4913330078125){
  s0+=0.04;
  s1+=0.96;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i0<2048.208984375){
  s0+=0.05128205128205128;
  s1+=0.9487179487179487;
 } else {
  s1+=1.0;
 }
}
if(i1<-3.088134765625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s1+=1.0;
 }
}
if(i2<45.5484619140625){
 s0+=1.0;
} else {
 if(i1<11.4210205078125){
  s0+=0.25;
  s1+=0.75;
 } else {
  s0+=0.01;
  s1+=0.99;
 }
}
if(i0<1407.433349609375){
 s0+=1.0;
} else {
 if(i2<2.02484130859375){
  s0+=1.0;
 } else {
  s0+=0.014492753623188406;
  s1+=0.9855072463768116;
 }
}
if(i0<1575.381591796875){
 if(i3<4138.9443359375){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i1<-2.8719482421875){
  s0+=1.0;
 } else {
  s0+=0.027522935779816515;
  s1+=0.9724770642201835;
 }
}
if(i0<1453.120849609375){
 s0+=1.0;
} else {
 if(i1<-1.58148193359375){
  s0+=1.0;
 } else {
  s0+=0.02654867256637168;
  s1+=0.9734513274336283;
 }
}
if(i2<2.02484130859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<2.02484130859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<-1.58148193359375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<45.5484619140625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.75;
  s1+=0.25;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.15;
  s1+=0.85;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.125;
  s1+=0.875;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i0<2048.208984375){
  s0+=0.043478260869565216;
  s1+=0.9565217391304348;
 } else {
  s1+=1.0;
 }
}
if(i2<45.5484619140625){
 s0+=1.0;
} else {
 if(i2<224.71023559570312){
  s0+=0.42857142857142855;
  s1+=0.5714285714285714;
 } else {
  s0+=0.00909090909090909;
  s1+=0.990909090909091;
 }
}
if(i1<-13.0201416015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1575.381591796875){
 if(i3<4138.9443359375){
  s0+=1.0;
 } else {
  s0+=0.16666666666666666;
  s1+=0.8333333333333334;
 }
} else {
 if(i0<1966.765625){
  s0+=0.22727272727272727;
  s1+=0.7727272727272727;
 } else {
  s1+=1.0;
 }
}
if(i2<2.02484130859375){
 s0+=1.0;
} else {
 if(i1<11.4210205078125){
  s0+=0.16666666666666666;
  s1+=0.8333333333333334;
 } else {
  s1+=1.0;
 }
}
if(i1<-3.088134765625){
 s0+=1.0;
} else {
 if(i2<224.71023559570312){
  s0+=0.2;
  s1+=0.8;
 } else {
  s0+=0.008547008547008548;
  s1+=0.9914529914529915;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.045454545454545456;
  s1+=0.9545454545454546;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.4;
  s1+=0.6;
 } else {
  s1+=1.0;
 }
}
if(i2<41.15252685546875){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.5;
  s1+=0.5;
 } else {
  s0+=0.008620689655172414;
  s1+=0.9913793103448276;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.08695652173913043;
  s1+=0.9130434782608695;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i1<31.94683837890625){
  s0+=0.038461538461538464;
  s1+=0.9615384615384616;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i1<11.52716064453125){
  s0+=0.4444444444444444;
  s1+=0.5555555555555556;
 } else {
  s1+=1.0;
 }
}
if(i2<45.5484619140625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.5;
  s1+=0.5;
 } else {
  s0+=0.009259259259259259;
  s1+=0.9907407407407407;
 }
}
if(i1<-5.7362060546875){
 s0+=1.0;
} else {
 if(i0<2048.208984375){
  s0+=0.05;
  s1+=0.95;
 } else {
  s1+=1.0;
 }
}
if(i0<1616.5074462890625){
 if(i3<4138.9443359375){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i1<-2.63043212890625){
  s0+=1.0;
 } else {
  s0+=0.009708737864077669;
  s1+=0.9902912621359223;
 }
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 if(i2<509.73468017578125){
  s0+=0.05;
  s1+=0.95;
 } else {
  s1+=1.0;
 }
}
if(i1<-4.44573974609375){
 s0+=1.0;
} else {
 if(i1<33.90179443359375){
  s0+=0.14285714285714285;
  s1+=0.8571428571428571;
 } else {
  s1+=1.0;
 }
}
if(i0<1599.311767578125){
 if(i3<4449.55712890625){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i0<1953.08154296875){
  s0+=0.17857142857142858;
  s1+=0.8214285714285714;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.25;
  s1+=0.75;
 } else {
  s0+=0.009009009009009009;
  s1+=0.990990990990991;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.42857142857142855;
  s1+=0.5714285714285714;
 } else {
  s0+=0.008928571428571428;
  s1+=0.9910714285714286;
 }
}
if(i2<45.5484619140625){
 s0+=1.0;
} else {
 if(i0<2048.208984375){
  s0+=0.1111111111111111;
  s1+=0.8888888888888888;
 } else {
  s1+=1.0;
 }
}
if(i1<-3.088134765625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s1+=1.0;
 }
}
if(i1<-5.7362060546875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i2<224.71023559570312){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s1+=1.0;
 }
}
if(i1<-12.77862548828125){
 s0+=1.0;
} else {
 if(i0<2048.208984375){
  s0+=0.04878048780487805;
  s1+=0.9512195121951219;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.125;
  s1+=0.875;
 } else {
  s1+=1.0;
 }
}
if(i2<45.5484619140625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<-3.088134765625){
 s0+=1.0;
} else {
 if(i0<1966.765625){
  s0+=0.03571428571428571;
  s1+=0.9642857142857143;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 if(i1<33.90179443359375){
  s0+=0.047619047619047616;
  s1+=0.9523809523809523;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.16666666666666666;
  s1+=0.8333333333333334;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1453.120849609375){
 s0+=1.0;
} else {
 if(i2<41.15252685546875){
  s0+=1.0;
 } else {
  s0+=0.008928571428571428;
  s1+=0.9910714285714286;
 }
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<-2.8719482421875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<45.5484619140625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<68.091064453125){
 s0+=1.0;
} else {
 if(i2<226.35107421875){
  s0+=0.5;
  s1+=0.5;
 } else {
  s0+=0.02459016393442623;
  s1+=0.9754098360655737;
 }
}
if(i1<-3.088134765625){
 s0+=1.0;
} else {
 if(i0<1966.765625){
  s0+=0.09523809523809523;
  s1+=0.9047619047619048;
 } else {
  s1+=1.0;
 }
}
if(i2<37.9146728515625){
 s0+=1.0;
} else {
 if(i1<10.42596435546875){
  s0+=0.4;
  s1+=0.6;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.84661865234375){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.08695652173913043;
  s1+=0.9130434782608695;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.3;
  s1+=0.7;
 } else {
  s1+=1.0;
 }
}
if(i2<63.69512939453125){
 s0+=1.0;
} else {
 if(i2<224.71023559570312){
  s0+=0.25;
  s1+=0.75;
 } else {
  s1+=1.0;
 }
}
if(i1<2.92034912109375){
 s0+=1.0;
} else {
 if(i1<31.94683837890625){
  s0+=0.16666666666666666;
  s1+=0.8333333333333334;
 } else {
  s1+=1.0;
 }
}
if(i0<1520.8087158203125){
 if(i3<4138.9443359375){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i0<1966.765625){
  s0+=0.28125;
  s1+=0.71875;
 } else {
  s1+=1.0;
 }
}
if(i1<-2.63043212890625){
 s0+=1.0;
} else {
 if(i1<31.94683837890625){
  s0+=0.05263157894736842;
  s1+=0.9473684210526315;
 } else {
  s1+=1.0;
 }
}
if(i2<2.02484130859375){
 s0+=1.0;
} else {
 if(i1<10.61090087890625){
  s0+=0.3333333333333333;
  s1+=0.6666666666666666;
 } else {
  s0+=0.017543859649122806;
  s1+=0.9824561403508771;
 }
}
if(i1<-5.7362060546875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i1<10.61090087890625){
 if(i3<4560.5205078125){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 s1+=1.0;
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i1<31.94683837890625){
  s0+=0.15789473684210525;
  s1+=0.8421052631578947;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 if(i2<498.2947998046875){
  s0+=0.09523809523809523;
  s1+=0.9047619047619048;
 } else {
  s1+=1.0;
 }
}
if(i2<42.31060791015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1453.120849609375){
 s0+=1.0;
} else {
 if(i1<-3.088134765625){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
}
if(i1<10.42596435546875){
 if(i3<4357.9658203125){
  s0+=1.0;
 } else {
  s1+=1.0;
 }
} else {
 s1+=1.0;
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}