project(superpiexl_crf)

find_package(OpenCV REQUIRED)

include_directories( include )

add_executable(superpixel_crf j2seg.cpp)

#target_link_libraries(superpixel_feature  /home/xduser/LiHuan/SuperPixel_Feature/lib/libjpeg.a /home/xduser/LiHuan/SuperPixel_Feature/lib/libSLIC.a /home/xduser/LiHuan/SuperPixel_Feature/lib/libopencv_lbp.a)
# link to libjpeg.a cause to imread() failed

target_link_libraries(superpixel_crf /home/xduser/LiHuan/superpixel_crf/lib/libSLIC.a /home/xduser/LiHuan/superpixel_crf/lib/libopencv_lbp.a)

#target_link_libraries(superpixel_crf ./lib/libSLIC.a /home/xduser/LiHuan/superpixel_crf/lib/libopencv_lbp.a)
# cannot find libSLIC.a

target_link_libraries(superpixel_crf ${OpenCV_LIBS} )

target_link_libraries(superpixel_crf /home/xduser/LiHuan/superpixel_crf/lib/libcnpy.a)

target_link_libraries(superpixel_crf /home/xduser/LiHuan/superpixel_crf/lib/libdensecrf.a)





