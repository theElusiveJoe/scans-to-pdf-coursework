#!/usr/bin/bash


for file in ../latex/content/geometry_images_monocrome/*;
do
  echo $file
  convert $file -monochrome $file
done
