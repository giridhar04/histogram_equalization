/*
Name: Giridhar Varuganti
Roll No: 19EC10070
OpenMP Assignment - 1
*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<unistd.h>
#include<omp.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"


unsigned char* to_grayscale(unsigned char *img, int height, int weight, int channels);

int main()
{   
    int width, height, channels;
    int n_threads;
    int no_of_bits = 8;
    int k = pow(2, no_of_bits)-1;
    int max_pixel = -1, min_pixel = INT_MAX;

    double wtime;

    char name[100];
    char temp[100];
    

    //Reading the image
    printf("Enter the name of png image (Don't mention extension type Eg: ""sample1""): ");
    scanf("%s", name);

    strcpy(temp, name);
    printf("\nLoading the image...\n");

    unsigned char* img = stbi_load(strcat(name, ".png"), &width, &height, &channels, 0);

    if(img == 0) {
        printf("Error loading image file\n");
        return -1;
    }
    printf("Image loaded with a width of %dpx, a height of %dpx and %d channels.\n", width, height, channels);
    

    //If the image is not in Grayscale convert it to Grayscale
    if(channels != 1){
        printf("\nImage is not in grayscale.\n");
        printf("Converting to grayscale...\n");
        img = to_grayscale(img, height, width, channels);
        
        if(img == 0){
            printf("Error in converting image to grayscale\n");
            return -1;
        }
        printf("Image converted to Grayscale.\n");
        channels = 1;
    }

    
    //Storing the image in a Matrix of size 
    int* arr[height];
    for (int i = 0; i < height; i++)
        arr[i] = (int*)malloc(width * sizeof(int));
 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++){

            arr[i][j] = img[i*width+j]; 
            if(arr[i][j] > max_pixel ) max_pixel = arr[i][j];
            if(arr[i][j] < min_pixel) min_pixel = arr[i][j];
        }
    } 
    printf("\nMin Pixel in the image = %d.\nMax Pixel in the image = %d.\n", min_pixel, max_pixel);




    //Histogram Equalization using OpenMP 
    printf ("\nNumber of processors available = %d.\n", omp_get_num_procs());
    printf ("Maximum number of threads = %d.\n", omp_get_max_threads());

    printf("\nEnter number of threads: \n");
    scanf("%d", &n_threads);

    
    /*Start of parallelization*/
    wtime = omp_get_wtime();
    printf("\n----Start of parallelization----\n");

    //Finding the frequency of each pixel
    long long int *freq = (long long int *) malloc( (k+1) * sizeof(long long int));
    #pragma omp parallel for num_threads(n_threads)
        for(int i=0; i<k+1; i++){
            freq[i] = 0;
        }

    #pragma omp parallel for num_threads(n_threads) schedule(static,1)
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                freq[arr[i][j]] ++;
            }
        }


    //Probability density function
    float *pdf = (float *) malloc( (k+1) * sizeof(float));
    #pragma omp parallel for num_threads(n_threads)
        for (int i=0; i<k+1; i++){
            pdf[i] = freq[i] / (float)(width*height);
        }


    //Transformation Function
    int *s = (int *) malloc( (k+1) * sizeof(int));
    #pragma omp parallel for num_threads(n_threads) schedule(static,1)
        for (int i=0; i<k+1; i++){
            float temp = 0;
            for(int j=0; j<=i; j++){
                temp = temp + (k)*pdf[j];
            }
            s[i] = round(temp);
        }

    
    //Mapping the results to original frequencies
    #pragma omp parallel for num_threads(n_threads) schedule(static,1)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++){
                img[i*width+j] = s[arr[i][j]];
            }
        }
    
    /*-----End of Parallelization----*/

    wtime = omp_get_wtime () - wtime;
    printf("Time taken = %fsec\n", wtime);
    printf("----End of parallelization----\n");
    
    printf("\nWriting Image...\n");
    stbi_write_png(strcat(temp, "_output.png"), width, height, 1, img, width*1);
    printf("Writing Image...Done!!!\n");


    stbi_image_free(img);
    
    return 0;
}


//Function to convert RGB image to Grayscale image
unsigned char *to_grayscale(unsigned char *img, int height, int width, int channels ){

    unsigned char *ptr = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    
    for(unsigned char *p = img, *q = ptr ; p != img + height*width*channels && q != ptr + height*width; p += channels, q += 1){
        double gamma= 1, gray;
        
        if(channels == 3){
            gray = 0.2126*pow(*(p), gamma) + 0.7152*pow(*(p+1), gamma) + 0.0722*pow(*(p+2), gamma);
        }
        else if(channels == 2){
            gray = 0.2126*pow(*(p), gamma) + 0.7152*pow(*(p+1), gamma);
        }
        else{
            gray = 0.2126*pow(*(p), gamma);
        }

        *q = gray;
    }

    return ptr;

}
